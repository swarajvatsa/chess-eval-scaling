"""
target.py — Turn a raw data row into the single number we train the brain on.

------------------------------------------------------------------------------
WHAT THIS FILE IS
------------------------------------------------------------------------------
Every row of our downloaded data describes one chess position and gives a verdict
on it as two possible fields:

    cp    : a centipawn score. 100 cp = "one pawn better". Can be None.
    mate  : a "mate in N" flag. e.g. 3 = mate in 3 moves. Can be None.
            (Exactly one of cp / mate is filled in on any given row.)

We cannot train directly on those raw fields. Three things have to happen first,
and this file is the ONE place they happen, so the rule is identical everywhere
(data prep, training, and later the C++ engine all import this logic). If the
conversion ever drifted between two places, the brain would be trained on one
scale and graded on another — a silent, poisonous bug. So: one function, one
truth, heavily commented.

------------------------------------------------------------------------------
THE THREE CONVERSIONS (in order)
------------------------------------------------------------------------------
1. POINT OF VIEW  (the rule we just PROVED in check_pov.py)
   The data's numbers are written from WHITE's side: "+ = good for White",
   no matter whose turn it is. But our brain always looks from the side that is
   ABOUT TO MOVE. So the target must be "+ = good for the mover". The fix:
       if White is to move:  keep the number as-is (the views already agree).
       if Black is to move:  FLIP THE SIGN (White's "+2" is the mover's "-2").
   This was verified on real data: among Black-to-move mate-in-1 positions,
   8129/8129 (100%) were written as a NEGATIVE mate — i.e. from White's side.

2. MATE -> A BIG CENTIPAWN NUMBER  (saturation)
   "Mate in 3" is not a centipawn score, but it is the best possible outcome,
   so we represent it as a huge positive score (and "being mated" as a huge
   negative one). We pick MATE_CP = 30000 — far larger than any normal material
   score, so the brain learns "mate beats everything" without us inventing a
   separate output just for mates. The exact size barely matters because the
   next step squashes it to "almost certainly winning" anyway.

3. CENTIPAWNS -> WIN PROBABILITY  (the squash)
   A raw centipawn score is unbounded and lopsided for learning: the difference
   between +300 and +400 cp matters a lot (could go either way), but the
   difference between +3000 and +3100 means nothing (both are dead winning). So
   we map centipawns to a win probability between 0 and 1 with the classic
   S-curve:
       win_prob = 1 / (1 + 10^(-cp / SCALE))      [a sigmoid in base 10-ish form]
   We use SCALE = 400, the long-standing chess convention (a 400-cp edge ~ a
   strong but not certain advantage). This is the SAME mapping the engine will
   use to interpret the brain's output, so training and play speak one language.
   The brain is then trained to predict this win probability (a smooth 0..1
   target), which behaves far better for learning than raw centipawns.

   Before squashing we CLAMP centipawns to +/- CP_CLAMP. The raw data ranges out
   to +/-20000 (near-mate blowouts); letting those through wastes the brain's
   capacity memorising the exact size of crushing positions it will squash to
   ~1.0 anyway. CP_CLAMP = 3000 keeps the meaningful middle and flattens the
   extremes — which is all the win-prob curve cares about.

------------------------------------------------------------------------------
WHAT WE HAND THE BRAIN
------------------------------------------------------------------------------
The brain trains on the WIN PROBABILITY (step 3 output, a number in [0,1]) from
the mover's point of view. We also expose the intermediate centipawn value
(steps 1-2, mover's POV) because the engine and some tests want it directly.
"""

import math

# --- constants (deliberately gathered in one place; do not scatter copies) ---

# A mate is stored as this many centipawns (positive = mover mates, negative =
# mover gets mated). Much bigger than any material score, smaller than overflow.
MATE_CP = 30000

# Raw centipawns are clamped to +/- this before squashing. Anything past here is
# "completely winning/losing" and the win-prob curve treats it as ~1 / ~0 anyway.
CP_CLAMP = 3000

# The centipawn-to-win-probability scale. 400 is the standard chess convention.
# This MUST match whatever the engine uses to read the brain's output later.
SCALE = 400.0


def target_centipawns(cp, mate, white_to_move):
    """
    Apply conversions 1 and 2: produce a centipawn score from the MOVER's point
    of view, with mates turned into a big saturated value.

    Inputs (straight from a data row):
        cp            : int or None  — centipawn score, WHITE's point of view.
        mate          : int or None  — mate-in-N, WHITE's point of view.
                                       (Exactly one of cp / mate is non-None.)
        white_to_move : bool         — True if it is White's turn in this position.

    Returns:
        int — centipawns from the MOVER's point of view, in [-MATE_CP, +MATE_CP].

    Step by step:
      a) Resolve the raw value into a single White's-POV centipawn number:
         - if `mate` is given: a positive mate (White mates) -> +MATE_CP,
           a negative mate (White gets mated) -> -MATE_CP.
         - else: take `cp`, clamped to the +/- CP_CLAMP window.
      b) Flip the sign if Black is to move (the proven point-of-view rule), so
         the result is finally from the mover's side.
    """
    # (a) collapse the row into one White's-POV centipawn value -----------------
    if mate is not None and mate != 0:
        # The sign of `mate` (White's POV) tells us who is delivering the mate.
        cp_white = MATE_CP if mate > 0 else -MATE_CP
    else:
        # Normal score. Guard against a None cp (shouldn't happen if mate is also
        # None, but we never want a crash mid-dataset) by treating it as level.
        raw = 0 if cp is None else int(cp)
        # Clamp the blowout tails into the window we actually care about.
        if raw > CP_CLAMP:
            raw = CP_CLAMP
        elif raw < -CP_CLAMP:
            raw = -CP_CLAMP
        cp_white = raw

    # (b) rotate from White's POV to the MOVER's POV ---------------------------
    # If White is moving, the two views already agree -> keep the number.
    # If Black is moving, White's "+" is the mover's "-" -> flip the sign.
    return cp_white if white_to_move else -cp_white


def win_probability(cp_mover_pov):
    """
    Apply conversion 3: squash a mover's-POV centipawn score into a win
    probability in (0, 1) using the standard chess logistic curve.

        win_prob = 1 / (1 + 10^(-cp / SCALE))

    Why this exact shape: a centipawn lead of `SCALE` (=400) maps to about 0.91,
    a dead-even position (0 cp) maps to exactly 0.5, and huge scores flatten
    toward 1.0 / 0.0. That flattening is the point — it stops the brain wasting
    effort distinguishing "winning" from "even more winning".

    Note we use 10**(...) (base 10) rather than e**(...); with SCALE=400 this is
    the familiar Elo-style logistic. The engine will use the identical formula to
    interpret the brain's raw output, so both ends agree on what a number means.
    """
    # math.pow handles the large/small exponents gracefully; the result is always
    # safely inside (0, 1) so there is no divide-by-zero or overflow to worry about.
    return 1.0 / (1.0 + math.pow(10.0, -cp_mover_pov / SCALE))


def target_win_probability(cp, mate, white_to_move):
    """
    The whole pipeline in one call: raw data row -> the win-probability target
    (mover's POV, in (0,1)) that the brain is actually trained to predict.

    This is the function the data-prep and training code should use. It simply
    chains the two stages above: get the mover's-POV centipawns, then squash.
    """
    return win_probability(target_centipawns(cp, mate, white_to_move))


# ---------------------------------------------------------------------------
# Self-test: run `python3 src/target.py` to sanity-check every rule by hand.
# These asserts encode, as executable examples, exactly what the file promises —
# so if anyone ever "simplifies" the sign rule the wrong way, this screams.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- point-of-view rule -------------------------------------------------
    # White to move, +200 cp (White better): mover is White, so target stays +200.
    assert target_centipawns(200, None, white_to_move=True) == 200
    # Black to move, +200 cp (data = White better): mover is Black, so it FLIPS
    # to -200 (the position is bad for the mover).
    assert target_centipawns(200, None, white_to_move=False) == -200
    # Black to move, -200 cp (data = Black better): flips to +200 (good for mover).
    assert target_centipawns(-200, None, white_to_move=False) == 200

    # --- mate handling ------------------------------------------------------
    # White to move, mate +3 (White mates): mover mates -> +MATE_CP.
    assert target_centipawns(None, 3, white_to_move=True) == MATE_CP
    # Black to move, mate -1 (data says White-POV negative => Black is mating):
    # flips to +MATE_CP, because the MOVER (Black) is the one delivering mate.
    # This is exactly the 8129/8129 case we verified in check_pov.py.
    assert target_centipawns(None, -1, white_to_move=False) == MATE_CP
    # White to move, mate -2 (White gets mated): -MATE_CP (bad for the mover).
    assert target_centipawns(None, -2, white_to_move=True) == -MATE_CP

    # --- clamp --------------------------------------------------------------
    # A blowout +20000 cp is clamped to +CP_CLAMP before anything else.
    assert target_centipawns(20000, None, white_to_move=True) == CP_CLAMP
    assert target_centipawns(-20000, None, white_to_move=True) == -CP_CLAMP

    # --- win-probability squash --------------------------------------------
    # Even position -> exactly 0.5.
    assert abs(win_probability(0) - 0.5) < 1e-9
    # A full SCALE (400 cp) edge -> about 0.909.
    assert abs(win_probability(SCALE) - 0.9090909) < 1e-4
    # Symmetry: a score and its negation are mirror images around 0.5.
    assert abs(win_probability(250) + win_probability(-250) - 1.0) < 1e-9
    # Mate squashes to essentially certain.
    assert win_probability(MATE_CP) > 0.999999

    # --- end-to-end ---------------------------------------------------------
    # Black to move, data mate -1 (Black mating) -> target win-prob ~ 1.0.
    assert target_win_probability(None, -1, white_to_move=False) > 0.999999
    # Black to move, data +300 (White better) -> mover is losing -> < 0.5.
    assert target_win_probability(300, None, white_to_move=False) < 0.5

    print("target.py self-test passed — POV flip, mate saturation, clamp, and "
          "win-prob squash all behave as specified.")
