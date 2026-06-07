"""
quantize_deep.py — Turn the trained DEEP residual judge into integer math + a
binary file, with a pure-integer golden evaluator the C++ must match bit-for-bit.

------------------------------------------------------------------------------
WHY THIS IS A SEPARATE FILE FROM quantize.py
------------------------------------------------------------------------------
The old judge was three plain layers, so quantize.py hard-codes "W0,b0,W1,b1,W2,b2".
The deep judge has a different shape: a first layer, then a VARIABLE number of
residual blocks, then a head. So it needs its own binary layout and its own golden
evaluator. The ARITHMETIC RULES, though, are exactly the same fixed-point scheme —
we deliberately designed the deep net (deepmodel.py) so every piece is something
the integer engine already does:

    a value v in [0,1]  is stored as the integer  V = round(v * ACT_MAX)   (0..ACT_MAX)
    a weight w          is stored as the integer  round(w * WEIGHT_SCALE)  (int16)

and every layer is "multiply-accumulate in a wide int, divide back by the scale,
round, clip". The ONLY genuinely new operation in the deep net is the residual
skip: after a block computes a delta D (an integer at the activation scale), we do
S' = clip(S + D, 0, ACT_MAX). That is a single integer add and a clip — nothing the
engine can't already do.

------------------------------------------------------------------------------
THE EXACT INTEGER RECIPE (this is the contract the C++ copies)
------------------------------------------------------------------------------
Let WS = WEIGHT_SCALE, AM = ACT_MAX. Activations live in 0..AM (representing [0,1]).

FIRST LAYER (sparse; inputs are 0/1 switches):
    acc[j]  = first_b[j]*1?  -> NO: first_b is stored *WS, inputs are 0/1, so
    acc[j]  = first_bq[j] + sum_{i active} first_Wq[j,i]          (scale = WS)
    S[j]    = clip(round(acc[j] * AM / WS), 0, AM)

EACH RESIDUAL BLOCK (s in 0..AM):
    # fc1: dim -> hidden, then clip
    a1[k]   = fc1_bq[k]*AM + sum_j fc1_Wq[k,j]*S[j]               (scale = WS*AM)
    H[k]    = clip(round(a1[k] * AM / (WS*AM)), 0, AM)   = clip(round(a1[k]/WS),0,AM)
    # fc2: hidden -> dim, NO activation -> this is the block's delta
    a2[j]   = fc2_bq[j]*AM + sum_k fc2_Wq[j,k]*H[k]               (scale = WS*AM)
    D[j]    = round(a2[j] * AM / (WS*AM))               = round(a2[j]/WS)   (signed)
    # residual skip + re-clip back into [0,1]
    S[j]    = clip(S[j] + D[j], 0, AM)

HEAD (dim -> 1, no activation):
    acc     = head_bq*AM + sum_j head_Wq[j]*S[j]                  (scale = WS*AM)
    raw     = round(acc / (WS*AM))        -> the integer centipawn-like score

All sums use a wide (int64 here, int32/int64 in C++) accumulator so nothing
overflows. We verify (below) that the worst-case accumulator stays well inside the
limit, and that the integer net matches the float net to a tiny error.

------------------------------------------------------------------------------
BINARY LAYOUT (little-endian), magic b"CHESSDP1"
------------------------------------------------------------------------------
    MAGIC        8 bytes   b"CHESSDP1"
    in_size      int32     768
    dim          int32     512    (width of the running signal / first-layer out)
    blocks       int32     N      (number of residual blocks)
    hidden       int32     512    (each block's internal width)
    weight_scale int32     256
    act_max      int32     255
    then int16 blocks in this exact order:
      first_W (dim x in_size)   first_b (dim)
      for each of N blocks:
        fc1_W (hidden x dim)    fc1_b (hidden)
        fc2_W (dim x hidden)    fc2_b (dim)
      head_W (1 x dim)          head_b (1)
"""

import argparse
import os
import struct

import numpy as np
import torch

from deepmodel import DeepJudge, SCALE, INPUT_SIZE, encode_board

MAGIC = b"CHESSDP1"
WEIGHT_SCALE = 256     # weights * this, rounded to int16 (matches shallow quantizer)
ACT_MAX = 255          # [0,1] activations represented as integers 0..255


def pick_weight_scale_deep(model):
    """Largest power-of-two WEIGHT_SCALE keeping every weight/bias inside int16.

    Hard-coding 256 overflows once a net is FULLY trained (its weights grow past
    ~128), giving "weight overflows int16". We measure the largest |weight| across
    first layer, all residual blocks, and the head, then drop the scale to the
    biggest power of two that still fits. The C++ deep loader reads the scale from
    the qbin header (eval_deep.cpp: weight_scale = hdr[4]), so any value stays
    bit-exact on the C++ side. Capped at 256 (the proven-good scale)."""
    amax = 0.0
    def upd(t):
        nonlocal amax
        amax = max(amax, float(np.abs(t.detach().numpy()).max()))
    upd(model.first.weight); upd(model.first.bias)
    for blk in model.tower:
        upd(blk.fc1.weight); upd(blk.fc1.bias); upd(blk.fc2.weight); upd(blk.fc2.bias)
    upd(model.head.weight); upd(model.head.bias)
    scale = 256
    while scale > 1 and amax * scale >= 32767:
        scale //= 2
    return scale


def _q(arr):
    """Quantize a float weight/bias array to int16 at WEIGHT_SCALE, asserting it
    fits int16 (so a too-large weight is caught here, not as silent corruption)."""
    qi = np.round(arr * WEIGHT_SCALE).astype(np.int64)
    amax = int(np.abs(qi).max()) if qi.size else 0
    assert amax < 32767, f"weight overflows int16 (absmax {amax}) at scale {WEIGHT_SCALE}"
    return qi.astype(np.int16), amax


def quantize_deep(model: DeepJudge):
    """Pull every layer out of the trained DeepJudge and quantize to int16. Returns
    a dict with first_W/first_b, per-block fc1/fc2 W&b, head_W/head_b, plus the
    dims. Also prints the per-tensor absmax so we can see headroom."""
    q = {"dim": model.dim, "blocks": model.blocks_n, "hidden": model.hidden}
    worst = 0

    fw, a = _q(model.first.weight.detach().numpy()); worst = max(worst, a)
    fb, a = _q(model.first.bias.detach().numpy());   worst = max(worst, a)
    q["first_W"], q["first_b"] = fw, fb

    q["blocks_w"] = []
    for i, blk in enumerate(model.tower):
        w1, a = _q(blk.fc1.weight.detach().numpy()); worst = max(worst, a)
        b1, a = _q(blk.fc1.bias.detach().numpy());   worst = max(worst, a)
        w2, a = _q(blk.fc2.weight.detach().numpy()); worst = max(worst, a)
        b2, a = _q(blk.fc2.bias.detach().numpy());   worst = max(worst, a)
        q["blocks_w"].append({"fc1_W": w1, "fc1_b": b1, "fc2_W": w2, "fc2_b": b2})

    hw, a = _q(model.head.weight.detach().numpy()); worst = max(worst, a)
    hb, a = _q(model.head.bias.detach().numpy());   worst = max(worst, a)
    q["head_W"], q["head_b"] = hw, hb

    # Worst-case accumulator headroom check (per output neuron): the biggest dense
    # sum is over `dim` (fc2 over hidden, fc1 over dim, head over dim) of
    # |weight| * AM. Make sure it stays well under int32's 2.1e9 (we accumulate in
    # int64 in the golden ref, and int32-lane SIMD in C++ — both safe here).
    span = max(model.dim, model.hidden)
    worst_acc = span * worst * ACT_MAX
    print(f"  quantized: per-tensor weight absmax {worst} "
          f"(int16 limit 32767); worst-case accumulator ~{worst_acc:,} "
          f"(int32 limit 2,147,483,647)", flush=True)
    assert worst_acc < 2_000_000_000, "accumulator could overflow int32 — lower WEIGHT_SCALE"
    return q


def _round_div(num, den):
    """Integer round-half-away-from-zero — the EXACT rule the C++ uses, so golden
    Python and C++ agree to the integer."""
    if num >= 0:
        return (num + den // 2) // den
    return -((-num + den // 2) // den)


def _clip_act(x):
    return 0 if x < 0 else (ACT_MAX if x > ACT_MAX else x)


def eval_int_deep(q, active_idx):
    """The GOLDEN integer evaluator for the deep judge — the arithmetic the C++
    must reproduce bit-for-bit. Returns the integer raw score (mover's POV)."""
    WS, AM = WEIGHT_SCALE, ACT_MAX
    S1 = WS * AM

    fW = q["first_W"].astype(np.int64)   # (dim, in)
    fb = q["first_b"].astype(np.int64)   # (dim,)

    # ---- first layer (sparse: sum active columns + bias, scale WS) ----
    acc = fb.copy()
    if active_idx:
        acc = acc + fW[:, active_idx].sum(axis=1)
    s = np.empty_like(acc)
    for j in range(acc.shape[0]):
        s[j] = _clip_act(_round_div(int(acc[j]) * AM, WS))

    # ---- residual blocks ----
    for blk in q["blocks_w"]:
        w1 = blk["fc1_W"].astype(np.int64); b1 = blk["fc1_b"].astype(np.int64)
        w2 = blk["fc2_W"].astype(np.int64); b2 = blk["fc2_b"].astype(np.int64)
        # fc1 + clip
        a1 = w1.dot(s) + b1 * AM
        h = np.empty_like(a1)
        for k in range(a1.shape[0]):
            h[k] = _clip_act(_round_div(int(a1[k]), WS))   # *AM/(WS*AM) == /WS
        # fc2 -> delta (no clip)
        a2 = w2.dot(h) + b2 * AM
        d = np.empty_like(a2)
        for j in range(a2.shape[0]):
            d[j] = _round_div(int(a2[j]), WS)
        # residual skip + re-clip
        for j in range(s.shape[0]):
            s[j] = _clip_act(int(s[j]) + int(d[j]))

    # ---- head ----
    hW = q["head_W"].astype(np.int64)    # (1, dim)
    hb = q["head_b"].astype(np.int64)    # (1,)
    acc2 = int(hW.dot(s)[0]) + int(hb[0]) * AM
    return _round_div(acc2, S1)


def write_binary_deep(q, path):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<6i", INPUT_SIZE, q["dim"], q["blocks"], q["hidden"],
                            WEIGHT_SCALE, ACT_MAX))
        np.ascontiguousarray(q["first_W"], dtype="<i2").tofile(f)
        np.ascontiguousarray(q["first_b"], dtype="<i2").tofile(f)
        for blk in q["blocks_w"]:
            for key in ("fc1_W", "fc1_b", "fc2_W", "fc2_b"):
                np.ascontiguousarray(blk[key], dtype="<i2").tofile(f)
        np.ascontiguousarray(q["head_W"], dtype="<i2").tofile(f)
        np.ascontiguousarray(q["head_b"], dtype="<i2").tofile(f)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deep_b4.pt")
    ap.add_argument("--out", default="models/deep_b4.qbin")
    ap.add_argument("--check-positions", type=int, default=3000)
    a = ap.parse_args()

    ck = torch.load(a.model, map_location="cpu")
    model = DeepJudge(dim=ck["dim"], blocks=ck["blocks"], hidden=ck["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded {a.model}  (dim={ck['dim']} blocks={ck['blocks']} "
          f"hidden={ck['hidden']}, val {ck.get('val_loss','?')})", flush=True)

    # Adaptively pick the weight scale so fully-trained nets don't overflow int16.
    # All quantizer functions read the module global WEIGHT_SCALE, so set it once
    # here; the chosen value is written into the qbin header and the C++ engine
    # reads it back (bit-exact regardless of value).
    global WEIGHT_SCALE
    WEIGHT_SCALE = pick_weight_scale_deep(model)
    print(f"  adaptive weight_scale = {WEIGHT_SCALE}", flush=True)

    q = quantize_deep(model)
    write_binary_deep(q, a.out)
    print(f"wrote {a.out}  ({os.path.getsize(a.out):,} bytes)  weight_scale={WEIGHT_SCALE}", flush=True)

    # ---- validate integer vs float over many varied positions ----
    import chess, random
    random.seed(0)
    floats, ints = [], []
    for _ in range(a.check_positions):
        b = chess.Board()
        for _ in range(random.randint(0, 40)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(random.choice(ms))
        with torch.no_grad():
            x = torch.from_numpy(encode_board(b)).unsqueeze(0)
            fwp = torch.sigmoid(model(x) / SCALE).item()
        # active indices in the same mover-POV order the encoder uses
        ai = [int(i) for i in np.nonzero(encode_board(b))[0]]
        raw = eval_int_deep(q, ai)
        iwp = 1.0 / (1.0 + np.exp(-raw / SCALE))
        floats.append(fwp); ints.append(iwp)

    floats = np.array(floats); ints = np.array(ints)
    mse = float(np.mean((floats - ints) ** 2))
    corr = float(np.corrcoef(floats, ints)[0, 1])
    maxerr = float(np.max(np.abs(floats - ints)))
    print(f"int-vs-float over {a.check_positions} positions:")
    print(f"  win-prob MSE {mse:.6e}  corr {corr:.6f}  max abs err {maxerr:.4f}")
    ok = corr > 0.999 and mse < 1e-4
    print(f"  verdict: {'PASS — integer deep brain is faithful' if ok else 'FAIL — investigate'}")


if __name__ == "__main__":
    main()
