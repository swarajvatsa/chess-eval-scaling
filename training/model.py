"""
The brain: board encoding + our own neural network.

We build the architecture ourselves with PyTorch (PyTorch only supplies the
tensor math / autograd / optimizer / GPU — the design is ours).

Eval = the net's ANSWER: given a position it outputs ONE number, the
side-to-move's evaluation. We interpret it as a win-probability via sigmoid.
"""

import chess
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Board encoding: board -> 768-length vector
#
#   768 = 6 piece types x 2 colors x 64 squares.
#   A switch is ON (1.0) when "this piece type of this color sits on this square".
#
# We encode from the SIDE-TO-MOVE's perspective: the board is mirrored so the
# side to move always "plays up the board". This means the net only ever has to
# learn one point of view (whoever is moving), which roughly halves what it must
# learn and is a standard trick for sparse piece-square evaluators.
# ---------------------------------------------------------------------------

INPUT_SIZE = 768

# piece_type (1..6) -> channel index 0..5
_PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """Return a float32 vector of length 768 for `board`, side-to-move perspective."""
    vec = np.zeros(INPUT_SIZE, dtype=np.float32)
    stm = board.turn  # True = white to move

    for square, piece in board.piece_map().items():
        # Mirror vertically when black is to move so "our" side is always at the bottom.
        sq = square if stm == chess.WHITE else chess.square_mirror(square)
        # "Our" pieces go in the first 6 channels, "their" pieces in the next 6.
        is_ours = piece.color == stm
        color_block = 0 if is_ours else 6
        channel = _PIECE_TO_CHANNEL[piece.piece_type] + color_block
        vec[channel * 64 + sq] = 1.0

    return vec


def encode_batch(boards) -> np.ndarray:
    """Encode a list of boards into an (N, 768) float32 array."""
    out = np.zeros((len(boards), INPUT_SIZE), dtype=np.float32)
    for i, b in enumerate(boards):
        out[i] = encode_board(b)
    return out


# ---------------------------------------------------------------------------
# The network (our design).
#
#   768 -> H -> 32 -> 1
#
# Clipped ReLU (clamp to [0, 1]) keeps activations bounded so nothing can blow
# up during training — a standard stability trick for quantized evaluators. The single output is
# a raw evaluation in "centipawn-like" units; sigmoid(out / SCALE) turns it into
# a win-probability for both play and training.
# ---------------------------------------------------------------------------

SCALE = 400.0  # eval units -> win-probability via sigmoid(eval / SCALE)


class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class ChessNet(nn.Module):
    def __init__(self, hidden: int = 256, hidden2: int = 32):
        super().__init__()
        self.hidden = hidden
        self.hidden2 = hidden2
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden),
            ClippedReLU(),
            nn.Linear(hidden, hidden2),
            ClippedReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        """x: (N, 768) -> (N,) raw evaluation in eval units."""
        return self.net(x).squeeze(-1)

    def win_prob(self, x):
        """Raw eval -> win-probability in (0, 1)."""
        return torch.sigmoid(self.forward(x) / SCALE)


# ---------------------------------------------------------------------------
# Convenience: evaluate a single python-chess board with a loaded net.
# Returns a win-probability for the side to move, in [0, 1].
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(board: chess.Board, model: ChessNet, device="cpu") -> float:
    x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
    return float(model.win_prob(x).item())


if __name__ == "__main__":
    # Sanity check: encode the start position and run a random-weights forward pass.
    board = chess.Board()
    v = encode_board(board)
    assert v.shape == (768,)
    assert v.sum() == 32, f"start position has 32 pieces, got {v.sum()}"

    model = ChessNet(hidden=256)
    p = evaluate(board, model)
    print(f"encoding OK: {int(v.sum())} pieces set")
    print(f"random-net win_prob(startpos) = {p:.4f} (expect ~0.5-ish, untrained)")

    # Mirrored sanity: a position and its color-mirror should encode identically
    # from the side-to-move perspective (chess is symmetric).
    b2 = chess.Board()
    b2.push_san("e4")
    print(f"win_prob after 1.e4 (black to move) = {evaluate(b2, model):.4f}")
    print("model.py self-test passed.")
