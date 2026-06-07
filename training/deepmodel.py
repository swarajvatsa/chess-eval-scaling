"""
deepmodel.py — The DEEP judge: a residual tower that can chain reasoning, designed
from the start so it survives the integer (quantized) C++ engine unchanged.

------------------------------------------------------------------------------
WHY A DEEP JUDGE AT ALL
------------------------------------------------------------------------------
Our old judge was two fat layers. We proved (arch_experiment.py) that a DEEPER
net with skip-connections, at the SAME knob-count, copies the depth-23 teacher
~12-21% more faithfully — because depth lets it CHAIN reasoning ("a trade opens
this file, SO the rook on it becomes strong"), which a wide-but-shallow net
cannot. This file is the shippable version of that idea.

------------------------------------------------------------------------------
THE ONE HARD CONSTRAINT: it must quantize to integers cleanly
------------------------------------------------------------------------------
The C++ engine runs the judge in pure integer math (no floating point) so it is
fast and bit-identical on every machine. That engine already knows ONE recipe:

    dense layer -> int32 accumulate -> divide by a fixed scale, round -> clip to
    [0, ACT_MAX] -> store as a small integer.

So every piece of THIS net is built only from that recipe plus an integer add.
We deliberately AVOID the things that make deep nets train nicely but are painful
in integers: no LayerNorm (needs float divide/sqrt), no unbounded activations
(would overflow the small integer range). Instead the "running signal" that flows
down the tower is kept BOUNDED in [0, 1] (it becomes [0, ACT_MAX] in integers) by
re-clipping after every block. That bounded highway is still a highway — the skip
connection still gives the training gradient a clean path to the bottom layers —
it is just a bounded one, which is exactly what the integer engine can hold.

------------------------------------------------------------------------------
SHAPE
------------------------------------------------------------------------------
    768 switches
       -> [FIRST LAYER]      Linear 768 -> DIM, clip to [0,1]      = the "summary"
       -> [RES BLOCK] x N     each: s = clip( s + W2( clip( W1 s ) ) )
       -> [HEAD]             Linear DIM -> 1                       = the eval number

The FIRST LAYER is the same sparse 768->DIM shape we already use, so the ~100x
incremental-accumulator trick still applies to it later (only the first layer is
sparse/incremental; the residual blocks are dense and run every eval). The eval
number is squashed by sigmoid(out/SCALE) into a win-probability, same convention
as everywhere else in the project.
"""

import chess
import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE = 768
SCALE = 400.0  # raw eval units -> win-probability via sigmoid(out / SCALE)


# ---------------------------------------------------------------------------
# Encoding — IDENTICAL to encode.py (mover's point of view, 768 switches). Kept
# here too so this module is self-contained for training. The C++ engine already
# has the bit-for-bit twin of this in eval.cpp::active_indices.
# ---------------------------------------------------------------------------
_PIECE_TO_CHANNEL = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                     chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}


def encode_board(board: chess.Board) -> np.ndarray:
    vec = np.zeros(INPUT_SIZE, dtype=np.float32)
    stm = board.turn
    for square, piece in board.piece_map().items():
        sq = square if stm == chess.WHITE else chess.square_mirror(square)
        color_block = 0 if piece.color == stm else 6
        channel = _PIECE_TO_CHANNEL[piece.piece_type] + color_block
        vec[channel * 64 + sq] = 1.0
    return vec


class ClippedReLU(nn.Module):
    """Clamp to [0, 1]. This is the activation the integer engine implements as
    'clip to [0, ACT_MAX]'. Using it everywhere (not just at the ends) is what
    keeps every intermediate value inside the small integer range the C++ side
    can hold — the whole net speaks the engine's language."""

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class ResBlock(nn.Module):
    """One reasoning step, built only from things the integer engine can do.

    The running signal `s` (already in [0,1]) goes through:
        h  = clip( W1 @ s + b1 )       # a thin bounded hidden activation
        d  = W2 @ h + b2               # the block's proposed change to the signal
        s' = clip( s + d )             # ADD it back (the skip) and re-bound to [0,1]

    The `s + d` is the skip-connection / 'wormhole': instead of replacing the
    signal, each block only nudges it, so the training gradient flows straight
    down the tower and the deep layers actually learn. The final clip is what
    keeps the highway bounded so the quantized engine never overflows. `hidden`
    is the block's internal width (its 'thinking room'); the signal width `dim`
    is unchanged so blocks can stack."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = ClippedReLU()
        # Start each block as a near-no-op: tiny output weights mean s' ~= s at
        # init, so a deep stack begins as a clean identity highway and learns its
        # nudges from there. This is what makes deep residual towers train stably.
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc2.weight, -1e-3, 1e-3)

    def forward(self, s):
        d = self.fc2(self.act(self.fc1(s)))
        return self.act(s + d)


class DeepJudge(nn.Module):
    """First layer -> N residual blocks -> single eval. `dim` is the width of the
    running signal (and the first-layer summary). `blocks` is how many reasoning
    steps it can chain. `hidden` is each block's internal width."""

    def __init__(self, dim=512, blocks=6, hidden=512):
        super().__init__()
        self.dim = dim
        self.blocks_n = blocks
        self.hidden = hidden
        self.first = nn.Linear(INPUT_SIZE, dim)
        self.first_act = ClippedReLU()
        self.tower = nn.Sequential(*[ResBlock(dim, hidden) for _ in range(blocks)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        s = self.first_act(self.first(x))
        s = self.tower(s)
        return self.head(s).squeeze(-1)

    def win_prob(self, x):
        return torch.sigmoid(self.forward(x) / SCALE)


@torch.no_grad()
def evaluate(board: chess.Board, model: DeepJudge, device="cpu") -> float:
    x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
    return float(model.win_prob(x).item())


if __name__ == "__main__":
    m = DeepJudge(dim=512, blocks=6, hidden=512)
    n = sum(p.numel() for p in m.parameters())
    b = chess.Board()
    x = torch.from_numpy(encode_board(b)).unsqueeze(0)
    print(f"DeepJudge dim=512 blocks=6 hidden=512 -> {n:,} params")
    print(f"start-position raw eval: {float(m(x)):.3f}  win_prob: {float(m.win_prob(x)):.3f}")
    # Sanity: bounded highway means every block output stays in [0,1].
    s = m.first_act(m.first(x))
    for i, blk in enumerate(m.tower):
        s = blk(s)
        assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0, f"block {i} escaped [0,1]"
    print("bounded-highway check passed — all block outputs in [0,1] (integer-safe).")
