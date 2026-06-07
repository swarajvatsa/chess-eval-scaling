"""
make_parity_deep.py — Emit (FEN, integer-eval) golden pairs for the DEEP judge.

Same idea as make_parity.py but for the deep residual judge: score thousands of
varied positions with the Python golden integer evaluator (quantize_deep.eval_int_deep)
and write "FEN<TAB>score" lines. The C++ parity_deep test loads the same .qbin and
must reproduce every integer exactly — proving the C++ deep port is faithful.
"""

import argparse
import random

import chess
import numpy as np
import torch

from deepmodel import DeepJudge, encode_board
from quantize_deep import quantize_deep, eval_int_deep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deep_b4.pt")
    ap.add_argument("--out", default="engine/parity_deep.txt")
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()

    ck = torch.load(a.model, map_location="cpu")
    m = DeepJudge(dim=ck["dim"], blocks=ck["blocks"], hidden=ck["hidden"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    q = quantize_deep(m)

    random.seed(1)
    rows = []
    while len(rows) < a.n:
        b = chess.Board()
        for _ in range(random.randint(0, 50)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(random.choice(ms))
        ai = [int(i) for i in np.nonzero(encode_board(b))[0]]
        score = eval_int_deep(q, ai)
        rows.append(f"{b.fen()}\t{score}")

    with open(a.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} deep parity rows -> {a.out}")


if __name__ == "__main__":
    main()
