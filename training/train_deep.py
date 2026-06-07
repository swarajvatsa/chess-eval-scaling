"""
train_deep.py — Train the integer-safe DEEP residual judge on the full dataset.

This is the production trainer for deepmodel.DeepJudge (the residual tower we
proved beats the wide-shallow net). It is the same fast, GPU-resident pipeline as
train_winprob.py — the whole packed dataset lives on the GPU, each step rebuilds a
dense batch with a scatter, win-probability MSE loss — only the MODEL changed.

Two things this trainer adds over the plain one, both because deep nets are
touchier to train than shallow ones:

  1. WARMUP + COSINE learning rate. A deep residual tower can diverge in the first
     few hundred steps if hit with the full learning rate cold. We ramp the LR up
     linearly for a short warmup, then cosine-decay it to ~0. This is standard for
     deep nets and costs nothing.

  2. GRADIENT CLIPPING. We cap the global gradient norm so a rare bad batch can't
     blow the weights up. Cheap insurance for an overnight unattended run.

It saves the best-by-validation checkpoint, storing the architecture (dim, blocks,
hidden) inside the file so the quantizer and any loader can rebuild it exactly.
"""

import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from deepmodel import DeepJudge, SCALE, INPUT_SIZE

PAD = INPUT_SIZE


def _load_shard(path):
    d = np.load(path)
    return d["idx"].astype(np.int16), d["target"].astype(np.float32)


def load_split(out_dir, split, device, max_shards):
    files = sorted(glob.glob(os.path.join(out_dir, f"{split}_*.npz")))
    if not files:
        raise SystemExit(f"no {split}_*.npz in {out_dir}")
    if max_shards > 0:
        files = files[:max_shards]
    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(min(mp.cpu_count(), len(files))) as pool:
        parts = pool.map(_load_shard, files)
    idx = np.concatenate([p[0] for p in parts])
    tgt = np.concatenate([p[1] for p in parts])
    n = idx.shape[0]
    print(f"  {split}: {n:,} positions ({idx.nbytes/1024**3:.1f} GiB) "
          f"in {time.time()-t0:.0f}s", flush=True)
    return (torch.from_numpy(idx).to(device),
            torch.from_numpy(tgt).to(device), n)


def densify(idx_batch, device):
    b = idx_batch.shape[0]
    dense = torch.zeros(b, INPUT_SIZE + 1, device=device)
    dense.scatter_(1, idx_batch.long(), 1.0)
    return dense[:, :INPUT_SIZE]


@torch.no_grad()
def val_loss(model, idx, tgt, device, bs):
    model.eval()
    n = idx.shape[0]
    total = 0.0
    for i in range(0, n, bs):
        xb = densify(idx[i:i+bs], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            wp = torch.sigmoid(model(xb) / SCALE)
            total += torch.sum((wp - tgt[i:i+bs]) ** 2).item()
    return total / n


def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("needs GPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("loading shards onto GPU...", flush=True)
    tr_idx, tr_tgt, n = load_split(a.shards, "train", device, a.max_shards)
    va_idx, va_tgt, nv = load_split(a.shards, "val", device, 0)

    model = DeepJudge(dim=a.dim, blocks=a.blocks, hidden=a.hidden).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"DeepJudge dim={a.dim} blocks={a.blocks} hidden={a.hidden} "
          f"({nparams:,} params)", flush=True)
    print(f"train {n:,} | val {nv:,} | epochs {a.epochs} | bs {a.batch_size} "
          f"| lr {a.lr} | warmup {a.warmup_steps}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.MSELoss()
    bs = a.batch_size
    nb = n // bs
    total_steps = nb * a.epochs

    def lr_at(step):
        # Linear warmup then cosine decay to ~0.
        if step < a.warmup_steps:
            return a.lr * (step + 1) / a.warmup_steps
        p = (step - a.warmup_steps) / max(1, total_steps - a.warmup_steps)
        return a.lr * 0.5 * (1 + math.cos(math.pi * p))

    step = 0
    best = float("inf")
    for epoch in range(1, a.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        for _ in range(nb):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            rows = torch.randint(0, n, (bs,), device=device)
            xb = densify(tr_idx[rows], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                wp = torch.sigmoid(model(xb) / SCALE)
                loss = loss_fn(wp, tr_tgt[rows])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            running += loss.item()
            step += 1
        vl = val_loss(model, va_idx, va_tgt, device, bs)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{a.epochs}  train {running/nb:.5f}  val {vl:.5f}  "
              f"{dt:.0f}s  lr {lr_at(step):.2e}", flush=True)
        if vl < best:
            best = vl
            torch.save({"state_dict": model.state_dict(),
                        "dim": a.dim, "blocks": a.blocks, "hidden": a.hidden,
                        "val_loss": vl, "epoch": epoch}, a.out)
    print(f"saved best (val {best:.5f}) -> {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--out", default="deep_b4.pt")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--warmup-steps", type=int, default=600)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-shards", type=int, default=0, help="0 = full dataset")
    a = ap.parse_args()
    train(a)


if __name__ == "__main__":
    main()
