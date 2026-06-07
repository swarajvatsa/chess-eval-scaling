"""
arch_experiment.py — Settle ONE question with numbers: is our judge too small/too
shallow to absorb the good data, and would going DEEPER (with skip-links) help?

------------------------------------------------------------------------------
WHY THIS SCRIPT EXISTS
------------------------------------------------------------------------------
We trained the judge on 292 million depth-23 positions and it plateaued at the
SAME held-out error (~0.0222) as the old judge on worse data. That is the
signature of a judge that is FULL: it has wrung out everything its knobs can
hold, so feeding it better data no longer moves the needle. The fix is not more
data — it is a judge with more (and better-arranged) capacity.

But "make it bigger" splits into two very different bets, and we refuse to guess:

  BET A — WIDTH: keep it 2 layers, just make those layers fatter (more knobs in
          the same shallow shape). Good at "add up material + per-square bonuses".

  BET B — DEPTH + SKIP-LINKS ("wormholes"): stack many thin layers so the net can
          chain reasoning ("this file WILL open BECAUSE of a trade, SO the rook is
          strong"), and add residual skip-connections so the training signal can
          still reach the bottom layers instead of dying on the way down.

This script trains both bets (plus the current baseline and a raw-capacity probe)
on the SAME data for the SAME number of epochs, and prints the held-out error of
each. If DEPTH wins at EQUAL knob-count, Bet B is proven and we commit to it. If
width ties depth, it was only ever about raw capacity (the cheaper fix). Either
way we get evidence, not a vibe.

------------------------------------------------------------------------------
WHAT WE MEASURE
------------------------------------------------------------------------------
Held-out (validation) mean-squared-error on the win-probability target — the same
honest number the main trainer reports. Lower = the judge copies the depth-23
teacher more faithfully = a stronger eval. We re-run the CURRENT architecture in
this same harness so every number is at the same budget (no comparing against an
old 40-epoch run).

NOTE ON ACTIVATIONS: the shipped judge uses a clamped activation (nice for the
integer/quantized engine). But clamping to [0,1] strangles a DEEP residual stream
(the skip-path needs to carry an unbounded running signal). Since this experiment
only has to ANSWER "does depth help", we train in plain float with ordinary ReLU
and pre-norm residual blocks — the arrangement that lets depth actually train. If
depth wins, solving "how to ship a deep judge fast" is the next problem, and a
good one to have.
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE = 768
SCALE = 400.0  # raw eval units -> win-probability via sigmoid(raw / SCALE)


# ===========================================================================
# DATA — load the packed shards onto the GPU exactly like the main trainer.
# ===========================================================================
def _load_shard(path):
    """Load one shard into (idx int16 (m,32), target f32 (m,)). int16 holds 0..768
    fine (well under 32767) and is half the bytes of int32, so the whole dataset
    fits in GPU memory."""
    d = np.load(path)
    return d["idx"].astype(np.int16), d["target"].astype(np.float32)


def load_split(out_dir, split, device, max_shards):
    """Read up to `max_shards` files of one split, stack them, move to the GPU.
    Reading the compressed .npz files is the slow part, so we fan it across CPU
    cores (one task per file) and then a cheap concatenate stitches them."""
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
    """Packed indices (B,32) -> dense 0/1 vectors (B,768). Scatter a 1 into each
    named column of a zero buffer; the extra column 768 is the PAD junk-bucket we
    slice off. Only the small batch is ever dense, so memory stays bounded."""
    b = idx_batch.shape[0]
    dense = torch.zeros(b, INPUT_SIZE + 1, device=device)
    dense.scatter_(1, idx_batch.long(), 1.0)
    return dense[:, :INPUT_SIZE]


# ===========================================================================
# ARCHITECTURES
# ===========================================================================
class MLP(nn.Module):
    """A plain stack of fully-connected layers — the WIDTH bet. Every input talks
    directly to every hidden unit; one or two fat hidden layers learn to add up
    material and per-square bonuses. Shallow, so it cannot chain multi-step
    reasoning, but it is cheap and trains easily. `widths` lists the hidden sizes,
    e.g. [1024, 256] is the judge we ship today."""

    def __init__(self, widths):
        super().__init__()
        layers = []
        prev = INPUT_SIZE
        for w in widths:
            layers += [nn.Linear(prev, w), nn.ReLU()]
            prev = w
        layers += [nn.Linear(prev, 1)]  # final scalar: the raw eval
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResBlock(nn.Module):
    """One 'wormhole' block — the heart of the DEPTH bet.

    A block transforms the running signal `x` through two thin linear layers and
    then ADDS the result back onto `x`:

        x_out = x + W2( relu( W1( norm(x) ) ) )

    The `+ x` is the skip-connection. It matters enormously for deep nets: it
    gives the training signal a clean highway straight to the bottom layers, so
    they keep learning instead of going silent (the reason naive deep stacks fail
    to train). `norm(x)` (LayerNorm, applied BEFORE the linears = 'pre-norm')
    keeps each block's input at a sane scale so stacking many of them stays stable.
    Stacking N of these lets the judge compose N steps of reasoning while the data
    only has to teach each thin step."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class ResNet(nn.Module):
    """The DEPTH + skip-link judge. First squeeze the 768 switches into a `dim`-wide
    running signal (the 'embedding'), then push it through `blocks` wormhole blocks
    that each refine it a little and add their refinement back on, then read out a
    single eval number. Depth = how many reasoning steps it can chain; the residual
    skips are what make that depth trainable."""

    def __init__(self, dim, blocks):
        super().__init__()
        self.embed = nn.Linear(INPUT_SIZE, dim)
        self.act = nn.ReLU()
        self.blocks = nn.Sequential(*[ResBlock(dim) for _ in range(blocks)])
        self.head_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.act(self.embed(x))
        x = self.blocks(x)
        return self.head(self.head_norm(x)).squeeze(-1)


def build(spec):
    """spec is ('mlp', [widths...]) or ('res', dim, blocks)."""
    if spec[0] == "mlp":
        return MLP(spec[1])
    return ResNet(spec[1], spec[2])


# ===========================================================================
# TRAIN ONE ARCHITECTURE
# ===========================================================================
@torch.no_grad()
def val_loss(model, idx, tgt, device, bs):
    """Mean-squared-error on held-out positions the net never trained on — the
    honest 'did it actually learn' number."""
    model.eval()
    n = idx.shape[0]
    total = 0.0
    for i in range(0, n, bs):
        xb = densify(idx[i:i+bs], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            wp = torch.sigmoid(model(xb) / SCALE)
            total += torch.sum((wp - tgt[i:i+bs]) ** 2).item()
    return total / n


def train_one(name, spec, data, args, device):
    tr_idx, tr_tgt, n, va_idx, va_tgt = data
    model = build(spec).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name}: {spec}  ({nparams:,} params) ===", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()
    bs = args.batch_size
    nb = n // bs
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        for _ in range(nb):
            # OOM-safe shuffle: draw bs random row ids (tiny) instead of a full
            # n-length permutation (which would be gigabytes every epoch).
            rows = torch.randint(0, n, (bs,), device=device)
            xb = densify(tr_idx[rows], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                wp = torch.sigmoid(model(xb) / SCALE)
                loss = loss_fn(wp, tr_tgt[rows])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        vl = val_loss(model, va_idx, va_tgt, device, bs)
        best = min(best, vl)
        print(f"  {name} epoch {epoch:2d}/{args.epochs}  train {running/nb:.5f}  "
              f"val {vl:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        if vl <= best:
            torch.save({"arch": spec, "state_dict": model.state_dict(),
                        "val_loss": vl, "epoch": epoch, "params": nparams},
                       os.path.join(args.outdir, f"arch_{name}.pt"))
    print(f"  >>> {name} BEST val {best:.5f}  ({nparams:,} params)", flush=True)
    return name, nparams, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="shards")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--max-shards", type=int, default=6,
                    help="how many train shards to load (0=all). 6 ~= 175M positions, "
                         "plenty to rank architectures while keeping epochs fast.")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--only", default="",
                    help="comma-separated subset of arch names to run (default: all). "
                         "Lets us re-run just the unfinished contestants without "
                         "repeating the ones already measured.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("needs GPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("loading data...", flush=True)
    tr_idx, tr_tgt, n = load_split(args.shards, "train", device, args.max_shards)
    va_idx, va_tgt, nv = load_split(args.shards, "val", device, 0)  # full val set
    data = (tr_idx, tr_tgt, n, va_idx, va_tgt)
    print(f"train {n:,} | val {nv:,} | epochs {args.epochs} | bs {args.batch_size}",
          flush=True)

    # The contestants. Width vs depth are deliberately matched on knob-count so the
    # comparison is fair; baseline reproduces today's judge in this harness; big_mlp
    # probes whether RAW capacity alone moves the floor.
    specs = [
        ("baseline", ("mlp", [1024, 256])),          # ~1.05M  — what we ship now
        ("wide",     ("mlp", [2048, 512])),          # ~2.62M  — the WIDTH bet
        ("deep",     ("res", 512, 4)),               # ~2.49M  — the DEPTH bet (matched)
        ("deeper",   ("res", 384, 10)),              # ~3.3M   — push depth further
        ("big_mlp",  ("mlp", [4096, 1024, 256])),    # ~7.5M   — raw-capacity ceiling
    ]

    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        specs = [(n, s) for (n, s) in specs if n in wanted]
        print(f"running subset: {[n for n,_ in specs]}", flush=True)

    results = []
    for name, spec in specs:
        results.append(train_one(name, spec, data, args, device))

    print("\n================ ARCHITECTURE SHOOTOUT ================", flush=True)
    print(f"{'name':10s} {'params':>12s} {'best_val':>10s}", flush=True)
    for name, p, v in sorted(results, key=lambda r: r[2]):
        print(f"{name:10s} {p:>12,} {v:>10.5f}", flush=True)
    print("lower val = judge copies the depth-23 teacher better = stronger eval",
          flush=True)


if __name__ == "__main__":
    main()
