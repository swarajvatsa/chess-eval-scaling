"""
train_winprob.py — Train the brain on the new shards (win-probability targets).

------------------------------------------------------------------------------
HOW THIS DIFFERS FROM THE OLD TRAINER
------------------------------------------------------------------------------
An earlier trainer stored each position as a dense 768-vector
plus TWO raw numbers (a reference-engine centipawn and a game result) and blended them
into a target at train time with a knob lambda. Our pipeline (build_dataset.py)
already did all of that work once, on the way in:

  - the position is stored COMPACTLY as up to 32 "on-switch" indices (uint16,
    padded to 32 with PAD=768), exactly the encode.py format, and
  - the target is already the final WIN PROBABILITY (a single float in 0..1,
    from the mover's point of view, via the proven target.py conversions).

So this trainer is simpler and more honest: it just learns to predict that stored
win probability. There is no lambda blend here (that knob comes back later, for
the self-play phase). It also reads a separate VALIDATION set and reports the
validation loss every epoch — that held-out number, not the training loss, is the
real measure of whether the brain is learning to generalise.

------------------------------------------------------------------------------
WHY IT IS FAST (same trick as before, kept)
------------------------------------------------------------------------------
The whole dataset lives on the GPU at once. ~291M positions x 32 indices x 2
bytes is ~18.6 GB, which fits the L4's 23 GB with room for the net and a batch.
Storing only the ~32 'on' indices (not all 768 numbers) is what makes that
possible. Each training step rebuilds the dense 768-wide batch on the GPU with a
cheap "scatter" (only the batch is ever dense, ~0.2 GB), so there is zero
CPU->GPU copying inside the loop and the GPU stays saturated.

------------------------------------------------------------------------------
WHAT THE BRAIN OUTPUTS AND HOW WE SCORE IT
------------------------------------------------------------------------------
The net outputs one raw number per position. We squash it with sigmoid(raw/SCALE)
into a win probability and compare it (mean-squared-error) against the stored
target win probability. SCALE=400 matches the convention used everywhere else, so
the brain's raw output stays interpretable as a centipawn-like score.
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn

from model import ChessNet, SCALE, INPUT_SIZE

PAD = INPUT_SIZE  # the dummy index 768 used to pad short positions


def _load_shard(path):
    """
    Worker: load one shard file into (idx int16 (m,32), target f32 (m,)).

    The shards store idx as uint16 (values 0..768), but we cast to int16 because
    PyTorch works smoothly with int16 and 768 fits comfortably below int16's max
    (32767). The dense rebuild later casts the batch to int64 only momentarily.
    """
    d = np.load(path)
    idx = d["idx"].astype(np.int16)          # (m, 32)
    target = d["target"].astype(np.float32)  # (m,)
    return idx, target


def load_split_to_gpu(out_dir, split, device, max_shards=0):
    """
    Load every shard of one split ('train' or 'val') and stack it onto the GPU.

    Returns (idx, target, n):
      idx    : (N, 32) int16 tensor on the GPU — the packed positions.
      target : (N,)    float32 tensor on the GPU — the win-probability targets.
      n      : number of positions.

    Loading is parallelised across CPU cores (one task per shard file) because
    decompressing/reading the .npz files is the slow part; stacking is cheap.

    max_shards (>0) caps how many shards of this split are read. On fleet boxes
    whose volume was restored from a snapshot, the FIRST read of each block is
    lazy-loaded from S3 at a throttled ~40 MB/s, so reading all ~21 train shards
    (~19 GB) cold can stall for many minutes. 8 shards (~230M positions) is
    plenty to rank nets by Elo, so capping the train split keeps the cold-read
    bounded. Matches train_deep.py's --max-shards.
    """
    files = sorted(glob.glob(os.path.join(out_dir, f"{split}_*.npz")))
    if not files:
        raise SystemExit(f"no {split}_*.npz shards in {out_dir}")
    if max_shards and max_shards > 0:
        files = files[:max_shards]

    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(min(mp.cpu_count(), len(files))) as pool:
        parts = pool.map(_load_shard, files)

    idx = np.concatenate([p[0] for p in parts])
    target = np.concatenate([p[1] for p in parts])
    n = idx.shape[0]
    gib = idx.nbytes / 1024**3
    print(f"  {split}: {n:,} positions  ({gib:.1f} GiB idx, loaded {time.time()-t0:.0f}s)",
          flush=True)

    idx = torch.from_numpy(idx).to(device)        # int16 (N,32)
    target = torch.from_numpy(target).to(device)  # f32 (N,)
    return idx, target, n


def densify(idx_batch, device):
    """
    Turn a batch of packed indices (B,32) into dense 0/1 vectors (B,768).

    We make a (B, 769) zero buffer, scatter a 1.0 into every column named by the
    indices (the extra column 768 is the PAD slot — a junk bucket), then slice it
    off so the net sees a clean (B,768). Casting to int64 (.long()) happens only
    here, on the small batch, so the full dataset stays int16 in memory.
    """
    b = idx_batch.shape[0]
    dense = torch.zeros(b, INPUT_SIZE + 1, device=device)
    dense.scatter_(1, idx_batch.long(), 1.0)
    return dense[:, :INPUT_SIZE]


@torch.no_grad()
def evaluate_val(model, idx, target, device, bs):
    """
    Compute mean-squared-error on the validation set (held-out positions the net
    never trained on). This is the honest measure of learning. We run it in
    batches so memory stays bounded, in eval mode, without gradients.
    """
    model.eval()
    n = idx.shape[0]
    total = 0.0
    for i in range(0, n, bs):
        xb = densify(idx[i:i + bs], device)
        tb = target[i:i + bs]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            wp = torch.sigmoid(model(xb) / SCALE)
            total += torch.sum((wp - tb) ** 2).item()
    return total / n


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("this trainer needs a GPU; no CUDA found")
    # Allow the GPU to use its fast lower-precision matmul paths.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("loading shards onto GPU...", flush=True)
    # Cap only the TRAIN split — val is small and we want a stable held-out set.
    tr_idx, tr_tgt, n = load_split_to_gpu(args.shards, "train", device,
                                          max_shards=args.max_shards)
    va_idx, va_tgt, nv = load_split_to_gpu(args.shards, "val", device)

    model = ChessNet(hidden=args.hidden, hidden2=args.hidden2).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"net: 768 -> {args.hidden} -> {args.hidden2} -> 1   ({nparams:,} params)", flush=True)
    print(f"train {n:,} | val {nv:,} | batch {args.batch_size} | epochs {args.epochs}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Cosine schedule: start at lr, smoothly decay to ~0 by the final epoch. Helps
    # the net settle into a good minimum instead of bouncing around at the end.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()
    bs = args.batch_size

    # How many full batches per epoch.
    nb = n // bs

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Shuffle WITHOUT a full-dataset permutation.
        #
        # The obvious `torch.randperm(n)` allocates an int64 array of length n
        # (~2.3 GiB for 289M rows) on the GPU EVERY epoch — that extra chunk is
        # what pushed us past the L4's 22 GiB. Instead we shuffle cheaply: pick a
        # random start offset and stride coprime-ish to n so consecutive epochs
        # visit positions in different orders, and draw each batch's rows with a
        # small random index tensor (only bs entries, ~0.5 MB). This gives plenty
        # of stochasticity for training without ever holding an n-length index.
        running = 0.0
        steps = 0
        t0 = time.time()
        for _b in range(nb):
            # Random batch of `bs` row ids in [0, n). randint allocates only bs
            # int64s (tiny), not n — so no multi-GiB spike. Sampling with
            # replacement across the epoch is standard and fine for SGD at this
            # scale (each row is seen ~once per epoch in expectation).
            rows = torch.randint(0, n, (bs,), device=device)
            xb = densify(tr_idx[rows], device)
            tb = tr_tgt[rows]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                wp = torch.sigmoid(model(xb) / SCALE)
                loss = loss_fn(wp, tb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            steps += 1
        sched.step()

        train_loss = running / max(steps, 1)
        val_loss = evaluate_val(model, va_idx, va_tgt, device, bs)
        dt = time.time() - t0
        seen = steps * bs  # positions visited this epoch
        print(f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.5f}  "
              f"val {val_loss:.5f}  {dt:.1f}s  ({seen/dt/1e6:.1f}M pos/s)", flush=True)

        # Save the checkpoint whenever validation improves — so we keep the BEST
        # net (by held-out loss), not just the last one. The checkpoint stores its
        # own layer widths so any loader can rebuild the exact architecture.
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(),
                        "hidden": args.hidden, "hidden2": args.hidden2,
                        "val_loss": val_loss, "epoch": epoch}, args.out)

    print(f"saved best (val {best_val:.5f}) -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards",
                    help="folder of train_*.npz / val_*.npz shards from build_dataset.py")
    ap.add_argument("--out", default="compact_champion.pt")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--hidden2", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-shards", type=int, default=0,
                    help="cap train shards loaded (0 = all); avoids cold-EBS stalls")
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    print(f"device: cuda | win-prob trainer | bs {args.batch_size}", flush=True)
    train(args)


if __name__ == "__main__":
    main()
