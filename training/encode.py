"""
encode.py — Turn a chess position into the list of numbers the brain reads.

------------------------------------------------------------------------------
WHAT "ENCODING" MEANS HERE
------------------------------------------------------------------------------
The brain is just a function from numbers to a number. It cannot look at a chess
board; it can only read a fixed-length list of numbers. So before the brain ever
sees a position, we have to translate the board into that list. That translation
is "encoding", and it MUST be done exactly the same way everywhere — when we
build training data, when we train, and later inside the C++ engine. If two
places encoded differently, the brain would be trained on one language and asked
to play in another. So this file is the ONE definition, kept free of any heavy
dependency (no torch) so everything can import it.

------------------------------------------------------------------------------
THE REPRESENTATION: 768 on/off switches
------------------------------------------------------------------------------
We describe a position with 768 switches, each either 0 (off) or 1 (on):

        768 = 6 piece kinds  x  2 colours  x  64 squares.

Read that as: "for every (piece-kind, colour, square) combination, is there
such a piece sitting on that square right now?" A starting position turns on
exactly 32 switches (the 32 pieces); everything else is 0. So although the list
is 768 long, only ~32 entries are ever 1 — it is mostly zeros. That sparsity is
something we exploit later (we store just the ~32 'on' positions instead of all
768 numbers).

------------------------------------------------------------------------------
THE ONE SUBTLE CHOICE: we look from the MOVER's side
------------------------------------------------------------------------------
We do NOT encode "White's pieces" and "Black's pieces". We encode "the pieces of
whoever is about to move" (first 6 kinds) and "the opponent's pieces" (next 6),
and we FLIP the board top-to-bottom when Black is to move so that the side to
move is always playing "up the board".

Why: chess is symmetric — a position with White to move is strategically the
same as the mirror-image position with Black to move. By always presenting the
board from the mover's side, the brain only has to learn ONE point of view
instead of two. That roughly halves what it must learn. (This is the exact same
"mover's point of view" choice that target.py relies on for the eval sign — the
board encoding and the training number agree on whose side we're on.)

------------------------------------------------------------------------------
HOW A SQUARE BECOMES AN INDEX
------------------------------------------------------------------------------
Squares are numbered 0..63 (a1=0, b1=1, ..., h8=63). When Black is to move we
mirror the square vertically (a1<->a8 etc.) with `square_mirror`, so "forward"
is the same direction for both sides from the brain's view. Each piece then maps
to one switch:

        channel  = piece_kind_index (0..5)  +  (0 if it's the mover's, else 6)
        switch   = channel * 64  +  (possibly-mirrored square)

That switch index (0..767) is set to 1. Collect all of them and you have the
position, in the brain's language.
"""

import chess
import numpy as np

# The list length. Named so other files refer to one constant, never a literal.
INPUT_SIZE = 768

# Map each piece kind to a channel 0..5. Order is arbitrary but FIXED forever:
# changing it would silently invalidate every trained brain.
_PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def active_indices(board: chess.Board):
    """
    Return the list of switch indices (each in 0..767) that are ON for `board`,
    from the MOVER's point of view. This is the compact form of the position:
    instead of 768 zeros-and-ones, just the handful (~32 max) of 'on' positions.

    Walk every occupied square. For each piece:
      - If Black is to move, mirror the square vertically so the mover always
        faces "up". (square_mirror(sq) swaps rank 1<->8, 2<->7, ...)
      - Decide whether the piece belongs to the mover ("ours", channels 0-5) or
        the opponent ("theirs", channels 6-11).
      - Compute channel*64 + square and add it to the list.

    The returned list is what we store in the dataset and what the brain reads.
    """
    stm = board.turn  # True = White to move, False = Black to move.
    out = []
    for square, piece in board.piece_map().items():
        # Mirror the square when Black moves, so "the mover's side" is always
        # at the bottom of the brain's mental board.
        sq = square if stm == chess.WHITE else chess.square_mirror(square)
        # First 6 channels = the mover's own pieces; next 6 = the opponent's.
        color_block = 0 if piece.color == stm else 6
        channel = _PIECE_TO_CHANNEL[piece.piece_type] + color_block
        out.append(channel * 64 + sq)
    return out


def encode_dense(board: chess.Board) -> np.ndarray:
    """
    Return the full 768-long float32 vector (mostly zeros, ~32 ones) for `board`.
    This is the "spelled-out" form: handy for a direct brain forward pass or a
    sanity check. Internally it just turns on the switches that active_indices
    reports. Most of our storage uses the compact index form instead, but the
    brain ultimately consumes this dense vector (we rebuild it cheaply per batch).
    """
    vec = np.zeros(INPUT_SIZE, dtype=np.float32)
    for idx in active_indices(board):
        vec[idx] = 1.0
    return vec


def encode_batch(boards) -> np.ndarray:
    """
    Encode a list of boards into one (N, 768) float32 array — convenience for
    evaluating or training on many positions at once.
    """
    out = np.zeros((len(boards), INPUT_SIZE), dtype=np.float32)
    for i, b in enumerate(boards):
        out[i] = encode_dense(b)
    return out


if __name__ == "__main__":
    # Quick self-check of the encoding's core promises.
    b = chess.Board()  # starting position
    idx = active_indices(b)
    assert len(idx) == 32, f"start position has 32 pieces, got {len(idx)}"
    assert all(0 <= i < INPUT_SIZE for i in idx), "an index escaped 0..767"
    assert encode_dense(b).sum() == 32, "dense vector should have 32 ones"

    # Mover's-POV symmetry: the start position with White to move, encoded, should
    # look identical to the same position with Black to move (chess is symmetric
    # and we always view from the mover). We fake 'black to move' by null-pushing.
    b2 = chess.Board()
    b2.push(chess.Move.null())  # now Black to move, same pieces
    assert set(active_indices(b)) == set(active_indices(b2)), \
        "mover's-POV encoding should be symmetric for the start position"

    print("encode.py self-test passed — 768 switches, mover's POV, 32 ones at start.")
