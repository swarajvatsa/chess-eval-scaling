"""
build_dataset.py — Turn the 316M-row parquet download into compact training shards.

------------------------------------------------------------------------------
WHAT THIS DOES AND WHY
------------------------------------------------------------------------------
On disk we have 10 parquet files (~316 million chess positions), each row a FEN
plus a reference-engine verdict (cp or mate, written from White's side). We cannot train
straight off parquet: the brain wants (a) the position already encoded as numbers
and (b) the verdict already converted to the win-probability target. Doing that
conversion 40 times (once per training epoch) would waste hours. So we do it ONCE
here and write the result in a form the GPU can load instantly and reuse forever.

For each surviving row we store three things:
    - the position, as up to 32 "on-switch" indices (the compact encoding) — we
      pad to exactly 32 with a dummy slot so every row is the same width;
    - the target win-probability (a single float in 0..1);
    - which split it belongs to (train or validation).

Storage math (why this is small): 292M rows x 32 indices x 2 bytes (uint16) =
~18.7 GB for the positions, plus 292M x 4 bytes = ~1.2 GB for the targets. That
fits comfortably and loads as a flat memory-mapped array — no parsing at train
time.

------------------------------------------------------------------------------
THE FILTERS AND CONVERSIONS (each row goes through this gauntlet)
------------------------------------------------------------------------------
1. DEPTH FILTER: keep only rows where the reference engine searched to depth >= 20. ~92% of
   the data clears this; the shallow rest is weaker signal we don't want.
2. PARSE the FEN into a board (skip the ~1-in-a-million malformed one).
3. SANITY: a legal position has <= 32 pieces; drop anything stranger.
4. TARGET: run (cp, mate, whose-turn) through target.py — which applies the
   proven White's-POV sign flip, mate saturation, clamp, and the win-prob squash.
5. ENCODE: run the board through encode.py to get its on-switch indices (already
   from the mover's point of view, matching the target).
6. SPLIT: decide train vs validation by HASHING THE FEN. Hashing (not random
   choice) means the same position always lands in the same split, so a position
   can never leak from train into validation — the val score stays honest.

------------------------------------------------------------------------------
PARALLELISM
------------------------------------------------------------------------------
The 10 files are independent, so we process them with one worker process each
(across the machine's cores). Every worker streams its file in row-batches
(bounded memory), runs the gauntlet, and appends to its own pair of output
shards (one train, one validation). At the end we write a manifest.json listing
every shard, its row count, and the column meaning — the trainer reads that to
know what to load. Workers never share memory, so there is no contention.
"""

import argparse
import glob
import hashlib
import json
import os
import time

import chess
import numpy as np
import pyarrow.parquet as pq

# Our own locked-down conversion + encoding (the single sources of truth).
from target import target_win_probability
from encode import active_indices, INPUT_SIZE

# Each stored position is padded to this many indices. A legal chess position has
# at most 32 pieces, so 32 slots always suffice. Unused slots get the PAD value.
MAX_PIECES = 32
# Dummy "switch" index used to pad short rows. It is 768 (one past the real range
# 0..767); the trainer knows to ignore this slot when rebuilding the position.
PAD = INPUT_SIZE

# Keep only positions the reference engine searched at least this deep.
MIN_DEPTH = 20

# Roughly 1 in this many positions is held out for validation (by FEN hash).
VAL_EVERY = 100  # ~1% validation


def split_of(fen: str) -> str:
    """
    Decide 'train' or 'val' for a position, deterministically from its FEN.

    We hash the FEN to a number and send ~1% of positions (those whose hash falls
    in one bucket) to validation. Because the decision depends only on the FEN, a
    given position ALWAYS goes to the same split no matter which file/worker sees
    it — so duplicates can't straddle the train/val line and inflate our val score.
    We hash only the board part of the FEN (first field) so positions that differ
    only in move-clock noise still split together.
    """
    board_part = fen.split(" ", 1)[0]
    h = hashlib.blake2b(board_part.encode("ascii"), digest_size=8).digest()
    n = int.from_bytes(h, "big")
    return "val" if (n % VAL_EVERY) == 0 else "train"


def process_file(args):
    """
    Worker: stream one parquet file through the gauntlet and write its shards.

    Receives (file_path, out_dir, worker_id, batch_rows, max_rows). Returns a
    small dict of per-file stats + the shard paths it wrote, which the parent
    collects into the manifest.
    """
    file_path, out_dir, worker_id, batch_rows, max_rows = args

    # Accumulators in memory, flushed to disk at the end of the file. We size them
    # generously and trim before saving. (One file is ~30M rows; its ~30M-row
    # arrays are a few hundred MB — fine per worker.)
    train_idx, train_tgt = [], []
    val_idx, val_tgt = [], []

    kept = skipped_depth = skipped_parse = skipped_pieces = 0
    seen = 0

    pf = pq.ParquetFile(file_path)
    # Iterate in row-batches so we never hold the whole file in memory at once.
    for batch in pf.iter_batches(batch_size=batch_rows,
                                 columns=["fen", "depth", "cp", "mate"]):
        fens = batch.column("fen").to_pylist()
        depths = batch.column("depth").to_pylist()
        cps = batch.column("cp").to_pylist()
        mates = batch.column("mate").to_pylist()

        for fen, depth, cp, mate in zip(fens, depths, cps, mates):
            seen += 1
            if max_rows and seen > max_rows:
                break

            # (1) depth filter
            if depth is None or depth < MIN_DEPTH:
                skipped_depth += 1
                continue

            # (2) parse
            try:
                board = chess.Board(fen)
            except Exception:
                skipped_parse += 1
                continue

            # (3) piece-count sanity
            idx = active_indices(board)
            if len(idx) > MAX_PIECES:
                skipped_pieces += 1
                continue

            # (4) target: convert verdict -> mover-POV win probability
            white_to_move = board.turn == chess.WHITE
            target = target_win_probability(cp, mate, white_to_move)

            # (5) pack the (<=32) indices into a fixed-width row padded with PAD
            row = np.full(MAX_PIECES, PAD, dtype=np.uint16)
            row[:len(idx)] = np.asarray(idx, dtype=np.uint16)

            # (6) route to train or validation by FEN hash
            if split_of(fen) == "val":
                val_idx.append(row)
                val_tgt.append(target)
            else:
                train_idx.append(row)
                train_tgt.append(target)
            kept += 1

        if max_rows and seen > max_rows:
            break

    # ---- write this worker's shards ----
    def save(split, idx_list, tgt_list):
        if not idx_list:
            return None
        X = np.stack(idx_list)                      # (n, 32) uint16
        y = np.asarray(tgt_list, dtype=np.float32)  # (n,)
        path = os.path.join(out_dir, f"{split}_{worker_id:02d}.npz")
        np.savez(path, idx=X, target=y)
        return {"path": path, "rows": int(X.shape[0])}

    train_shard = save("train", train_idx, train_tgt)
    val_shard = save("val", val_idx, val_tgt)

    return {
        "worker": worker_id,
        "file": os.path.basename(file_path),
        "seen": seen,
        "kept": kept,
        "skipped_depth": skipped_depth,
        "skipped_parse": skipped_parse,
        "skipped_pieces": skipped_pieces,
        "train_shard": train_shard,
        "val_shard": val_shard,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/positions",
                    help="folder of input parquet files")
    ap.add_argument("--out-dir", default="shards",
                    help="folder to write training shards + manifest into")
    ap.add_argument("--batch-rows", type=int, default=200_000,
                    help="parquet rows read per batch (bounds per-worker memory)")
    ap.add_argument("--max-rows-per-file", type=int, default=0,
                    help="cap rows scanned per file (0 = all). Use a small value for a quick smoke test.")
    ap.add_argument("--workers", type=int, default=None,
                    help="number of worker processes (default: one per input file)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files in {args.data_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    tasks = [
        (f, args.out_dir, wid, args.batch_rows, args.max_rows_per_file)
        for wid, f in enumerate(files)
    ]

    print(f"building dataset from {len(files)} files -> {args.out_dir}", flush=True)
    print(f"filters: depth>={MIN_DEPTH}, <= {MAX_PIECES} pieces; ~{100//VAL_EVERY}% held out for val",
          flush=True)
    t0 = time.time()

    # One process per file. (If --workers given, the pool size is capped to it,
    # and files queue for the next free slot.)
    import multiprocessing as mp
    nproc = args.workers or len(files)
    with mp.Pool(nproc) as pool:
        results = pool.map(process_file, tasks)

    # ---- assemble manifest + totals ----
    manifest = {
        "format": "v1",
        "encoding": "768 mover-POV on-switch indices, padded to 32 with PAD=768",
        "target": "mover-POV win probability in (0,1) via target.py",
        "min_depth": MIN_DEPTH,
        "max_pieces": MAX_PIECES,
        "pad_index": PAD,
        "train_shards": [],
        "val_shards": [],
    }
    tot = dict(seen=0, kept=0, skipped_depth=0, skipped_parse=0, skipped_pieces=0,
               train_rows=0, val_rows=0)
    for r in results:
        for k in ("seen", "kept", "skipped_depth", "skipped_parse", "skipped_pieces"):
            tot[k] += r[k]
        if r["train_shard"]:
            manifest["train_shards"].append(r["train_shard"])
            tot["train_rows"] += r["train_shard"]["rows"]
        if r["val_shard"]:
            manifest["val_shards"].append(r["val_shard"])
            tot["val_rows"] += r["val_shard"]["rows"]
    manifest["totals"] = tot

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    dt = time.time() - t0
    print("\n==== DONE ====", flush=True)
    print(f"  scanned       : {tot['seen']:,}")
    print(f"  kept          : {tot['kept']:,}   (train {tot['train_rows']:,} | val {tot['val_rows']:,})")
    print(f"  dropped depth : {tot['skipped_depth']:,}")
    print(f"  dropped parse : {tot['skipped_parse']:,}")
    print(f"  dropped pieces: {tot['skipped_pieces']:,}")
    print(f"  time          : {dt/60:.1f} min")
    print(f"  manifest      : {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
