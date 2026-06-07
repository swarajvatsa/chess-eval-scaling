"""
make_parity.py — Emit (FEN, integer-eval) pairs as the golden reference for C++.

We generate a few thousand varied positions, score each with the Python golden
integer evaluator (quantize.eval_int), and write "FEN<TAB>score" lines. The C++
parity test loads the same .qbin, evaluates each FEN, and must produce the EXACT
same integer for every line. That proves the C++ port of the brain is faithful.
"""

import argparse
import random

import chess
import torch

from model import ChessNet
from encode import active_indices
from quantize import quantize_net, eval_int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="compact_champion.pt")
    ap.add_argument("--out", default="engine/parity.txt")
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()

    ck = torch.load(a.model, map_location="cpu")
    m = ChessNet(hidden=ck["hidden"], hidden2=ck["hidden2"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    q = quantize_net(m)

    random.seed(1)
    rows = []
    while len(rows) < a.n:
        b = chess.Board()
        for _ in range(random.randint(0, 50)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(random.choice(ms))
        score = eval_int(q, active_indices(b))
        # Use a full 6-field FEN so the C++ parser (which reads 4 fields) is happy.
        rows.append(f"{b.fen()}\t{score}")

    with open(a.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} parity rows -> {a.out}")


if __name__ == "__main__":
    main()
