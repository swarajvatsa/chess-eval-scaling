"""
quantize.py — Convert the trained float brain into integer math + a binary file.

------------------------------------------------------------------------------
WHY QUANTIZE AT ALL
------------------------------------------------------------------------------
The brain was trained in floating point (real numbers like 0.4189). Floating-
point math is flexible but comparatively slow, especially when we call the brain
millions of times per move inside a deep search. CPUs are dramatically faster at
INTEGER arithmetic done in bulk (a single SIMD instruction can add 16 or 32
integers at once). So before the engine ships in C++, we convert every weight in
the brain from a real number into a small integer, with a fixed "scale" so we can
convert back. This is "quantization". Done right, the integer brain gives almost
exactly the same answers as the float brain but runs many times faster.

------------------------------------------------------------------------------
THE SCALING IDEA (fixed-point integers)
------------------------------------------------------------------------------
A weight like 0.4189 becomes an integer by multiplying by a fixed scale and
rounding: with scale 64, 0.4189 -> round(0.4189 * 64) = 27. To use it we divide
back out by 64. The bigger the scale, the finer the precision — but the integer
must still fit its box (we use int16, which holds -32768..32767). We measured the
trained net's weight ranges to pick scales that are both precise AND safe:

   layer weights absmax were ~7.9, ~17.4, ~108 (per layer).
   With WEIGHT_SCALE = 64:  108 * 64 = 6912  -> fits int16 easily. Good.

The board inputs are 0/1 switches (no scaling needed). The hidden activations use
a "clipped relu" that squashes values into [0,1]; we represent that range as the
integers 0..ACT_MAX (=127), i.e. activation scale 127. So after each layer we:
   - multiply inputs by weights and sum (in a WIDE int32 accumulator so the sum
     can't overflow),
   - divide back by the appropriate scale,
   - clip into [0, ACT_MAX] for the next layer.

------------------------------------------------------------------------------
WHAT WE WRITE TO DISK
------------------------------------------------------------------------------
A single binary file the C++ engine will read. Layout (all little-endian):

   MAGIC      8 bytes   b"CHESSBR1"  (so the loader can sanity-check the file)
   in_size    int32     768
   hidden     int32     1024
   hidden2    int32     256
   weight_scale int32   64
   act_max    int32     127
   then the weights, each layer as int16, in this exact order:
     W0 (hidden x in_size)   b0 (hidden)
     W1 (hidden2 x hidden)   b1 (hidden2)
     W2 (1 x hidden2)        b2 (1)

The C++ loader will read the header, then read each block by its known size. We
also provide a pure-Python integer evaluator here (eval_int) that does the exact
same arithmetic the C++ will do — this is the GOLDEN REFERENCE the C++ must match
bit-for-bit, and the thing we validate the quantization against right now.

------------------------------------------------------------------------------
HOW WE VALIDATE
------------------------------------------------------------------------------
We run thousands of real positions through BOTH the float brain and the integer
brain and check the win-probabilities agree very closely (correlation ~1.0, tiny
error). If they do, the integer version is faithful and safe to ship. If not, we
raise the scales or investigate before building any C++.
"""

import argparse
import struct

import numpy as np
import torch

from model import ChessNet, SCALE
from encode import encode_dense, INPUT_SIZE

MAGIC = b"CHESSBR1"

# Fixed-point scales (chosen from the measured weight ranges; see module docstring).
#
# WEIGHT_SCALE = 256: weights are multiplied by this then rounded to int16. The
# output layer's weights are large (absmax ~108), and at the old scale of 64 the
# rounding of those 256 large weights threw the raw score off by ~20 (87 vs 67),
# enough to fail the fidelity gate. At 256 the worst weight is 108*256=27776, still
# safely inside int16's 32767 ceiling, and 4x more precise.
WEIGHT_SCALE = 256
# clipped-relu activations as integers 0..ACT_MAX. 255 keeps the layer-1/2 int32
# accumulators safe with WEIGHT_SCALE=256 (worst case ~1.8e9 < 2.1e9) while giving
# the activations 8 bits of resolution.
ACT_MAX = 255


def pick_weight_scale(model):
    """
    Choose the largest power-of-two WEIGHT_SCALE that keeps EVERY quantized weight
    and bias inside int16 (|x*scale| < 32767).

    The old code hard-coded 256, tuned to the lightly-trained champion whose largest
    output weight was ~108 (108*256=27648, just under the ceiling). A FULLY trained
    net grows those output weights past ~128, so 256 overflows int16 ("W2 overflows
    int16"). Rather than guess, we measure the largest |weight| across all layers and
    drop the scale to the biggest power of two that still fits. Powers of two keep the
    integer division in eval_int exact and cheap. The C++ engine reads the scale from
    the qbin header (eval.cpp: weight_scale = hdr[3]), so any value here stays
    bit-exact on the C++ side — we are not hard-coding a contract.
    """
    layers = [l for l in model.net if hasattr(l, "weight")]
    amax = 0.0
    for layer in layers:
        amax = max(amax, float(np.abs(layer.weight.detach().numpy()).max()))
        amax = max(amax, float(np.abs(layer.bias.detach().numpy()).max()))
    # Largest power of two s.t. amax*scale < 32767. Cap at the original 256 (no need
    # to go finer than the proven-good scale; more precision buys nothing downstream).
    scale = 256
    while scale > 1 and amax * scale >= 32767:
        scale //= 2
    return scale


def quantize_net(model, weight_scale=None):
    """
    Pull the three layers out of the trained float model and turn each weight and
    bias into a rounded int16 (multiplying by weight_scale). Returns (dict, scale)
    where the dict holds int16 numpy arrays — the exact numbers we will write to
    disk and feed the integer evaluator.

    weight_scale defaults to an adaptive value (pick_weight_scale) so any net,
    however large its trained weights, quantizes without overflowing int16.
    """
    if weight_scale is None:
        weight_scale = pick_weight_scale(model)
    layers = [l for l in model.net if hasattr(l, "weight")]
    assert len(layers) == 3, "expected exactly 3 linear layers"

    out = {}
    names = [("W0", "b0"), ("W1", "b1"), ("W2", "b2")]
    for (wn, bn), layer in zip(names, layers):
        w = layer.weight.detach().numpy()           # (out, in)
        b = layer.bias.detach().numpy()             # (out,)
        wq = np.round(w * weight_scale).astype(np.int32)
        bq = np.round(b * weight_scale).astype(np.int32)
        assert np.abs(wq).max() < 32767, f"{wn} overflows int16 at scale {weight_scale}"
        assert np.abs(bq).max() < 32767, f"{bn} overflows int16 at scale {weight_scale}"
        out[wn] = wq.astype(np.int16)
        out[bn] = bq.astype(np.int16)
    return out, weight_scale


def _round_div(num, den):
    """Integer round-half-away-from-zero division — the EXACT rule the C++ uses,
    so the golden Python eval and the C++ eval agree to the integer. (numpy.round
    rounds half-to-even, which would occasionally differ by 1; we avoid that by
    doing pure integer math here, identical on both sides.)"""
    if num >= 0:
        return (num + den // 2) // den
    return -((-num + den // 2) // den)


def eval_int(q, active_idx, weight_scale=WEIGHT_SCALE):
    """
    The GOLDEN integer evaluator — the EXACT integer arithmetic the C++ copies.
    Returns an INTEGER raw score (centipawn-like), from the mover's POV.

    All math is pure integer (no floats), so it is reproducible bit-for-bit in any
    language. Because the board inputs are 0/1, layer 0 is just "sum the active
    weight columns + bias" — no multiplies.

      Layer 0: acc0[j] = b0[j] + sum_{i active} W0[j,i]          (scale = WS)
               a0[j]   = clip(round(acc0[j]*ACT_MAX / WS), 0, ACT_MAX)
      Layer 1: acc1[k] = b1[k]*ACT_MAX + sum_j W1[k,j]*a0[j]     (scale = WS*ACT_MAX)
               a1[k]   = clip(round(acc1[k]*ACT_MAX / (WS*ACT_MAX)), 0, ACT_MAX)
      Layer 2: acc2    = b2[0]*ACT_MAX + sum_k W2[k]*a1[k]
               raw     = round(acc2 / (WS*ACT_MAX))   -> the integer score
    """
    W0, b0 = q["W0"].astype(np.int64), q["b0"].astype(np.int64)
    W1, b1 = q["W1"].astype(np.int64), q["b1"].astype(np.int64)
    W2, b2 = q["W2"].astype(np.int64), q["b2"].astype(np.int64)
    WS, AM = weight_scale, ACT_MAX
    S1 = WS * AM

    # --- Layer 0 ---
    acc0 = b0.copy()
    if active_idx:
        acc0 = acc0 + W0[:, active_idx].sum(axis=1)
    a0 = np.empty_like(acc0)
    for j in range(acc0.shape[0]):
        q0 = _round_div(int(acc0[j]) * AM, WS)
        a0[j] = 0 if q0 < 0 else (AM if q0 > AM else q0)

    # --- Layer 1 ---
    acc1 = W1.dot(a0) + b1 * AM
    a1 = np.empty_like(acc1)
    for k in range(acc1.shape[0]):
        q1 = _round_div(int(acc1[k]) * AM, S1)
        a1[k] = 0 if q1 < 0 else (AM if q1 > AM else q1)

    # --- Layer 2 ---
    acc2 = int(W2.dot(a1)[0]) + int(b2[0]) * AM
    return _round_div(acc2, S1)   # integer raw score


def write_binary(q, path, weight_scale=WEIGHT_SCALE):
    """Write the quantized weights to the binary file the C++ engine will read,
    in the exact header+blocks layout described in the module docstring. The
    chosen weight_scale goes into the header so the C++ engine uses the SAME scale
    (it reads hdr[3]); this is what lets the scale be adaptive per net."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<5i", INPUT_SIZE, q["W0"].shape[0], q["W1"].shape[0],
                            weight_scale, ACT_MAX))
        for key in ("W0", "b0", "W1", "b1", "W2", "b2"):
            # tofile writes little-endian int16 on x86; we keep arrays C-contiguous.
            np.ascontiguousarray(q[key], dtype="<i2").tofile(f)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="compact_champion.pt")
    ap.add_argument("--out", default="models/compact_champion.qbin")
    ap.add_argument("--check-positions", type=int, default=3000,
                    help="how many random positions to validate int-vs-float on")
    a = ap.parse_args()

    ck = torch.load(a.model, map_location="cpu")
    model = ChessNet(hidden=ck["hidden"], hidden2=ck["hidden2"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded {a.model}  ({ck['hidden']}->{ck['hidden2']})", flush=True)

    q, ws = quantize_net(model)
    write_binary(q, a.out, ws)
    import os
    print(f"wrote {a.out}  ({os.path.getsize(a.out):,} bytes)  weight_scale={ws}", flush=True)

    # ---- validate: integer eval must closely match float eval ----
    # Generate varied positions by playing random legal moves from the start.
    import chess, random
    from encode import active_indices
    random.seed(0)
    floats, ints = [], []
    for _ in range(a.check_positions):
        b = chess.Board()
        for _ in range(random.randint(0, 40)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(random.choice(ms))
        # float win-prob from the real net
        with torch.no_grad():
            x = torch.tensor(encode_dense(b)).unsqueeze(0)
            fwp = torch.sigmoid(model(x) / SCALE).item()
        # integer win-prob from the golden integer eval (same scale as the qbin)
        raw = eval_int(q, active_indices(b), ws)
        iwp = 1.0 / (1.0 + np.exp(-raw / SCALE))
        floats.append(fwp); ints.append(iwp)

    floats = np.array(floats); ints = np.array(ints)
    mse = float(np.mean((floats - ints) ** 2))
    corr = float(np.corrcoef(floats, ints)[0, 1])
    maxerr = float(np.max(np.abs(floats - ints)))
    print(f"int-vs-float over {a.check_positions} positions:")
    print(f"  win-prob  MSE {mse:.6e}   correlation {corr:.6f}   max abs err {maxerr:.4f}")
    ok = corr > 0.999 and mse < 1e-4
    print(f"  verdict: {'PASS — integer brain is faithful' if ok else 'FAIL — raise scales / investigate'}")


if __name__ == "__main__":
    main()
