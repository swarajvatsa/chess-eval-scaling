# Training

The Python side: turn a corpus of labelled chess positions into a trained
position evaluator, then quantize it to integer weights the C++ engine can play
with. **Train in Python, ship in C++** — the bridge is a `.qbin` file plus a
bit-exact parity check.

```bash
pip install -r requirements.txt
```

A single GPU with ~24 GB is comfortable; CPU training works but is slow. The
dataset is held in GPU memory as integers and densified per batch.

## 1. Build the dataset

```bash
python build_dataset.py --parquet data/positions --out shards/
```

Inputs are positions labelled with a reference engine's evaluation (centipawns or
mate, plus the search depth). The builder:

- **keeps** only positions searched to depth ≥ 20 (the rest is weaker signal),
- **encodes** each board as up to 32 active indices in a 768-feature
  piece-square space, from the side-to-move's perspective,
- **converts** the evaluation to the training target — a **win probability** (see
  `target.py`): collapse to the mover's point of view, saturate mates, clamp
  extreme scores, then squash centipawns through `1 / (1 + 10^(-cp/400))`,
- **splits** train/validation by a hash of the position (leakage-free),

and writes `train_*.npz` / `val_*.npz` shards plus a manifest.

## 2. Train a network

Compact fully-connected net:

```bash
python train.py --shards shards --out compact_champion.pt \
  --hidden 1024 --hidden2 256 --epochs 30 --batch-size 65536 --lr 2e-3
```

Deep residual-tower net:

```bash
python train_deep.py --shards shards --out deep_b4.pt \
  --dim 512 --blocks 4 --hidden 512 --epochs 30 --batch-size 32768 \
  --lr 5e-4 --warmup-steps 600 --grad-clip 1.0
```

The loss is mean-squared error in win-probability space, which is why validation
losses look small (~0.02–0.03). Use `--max-shards N` to train on a subset for a
quick run (0 = all data).

> **Note on deep nets:** towers of ≥ 6 residual blocks can diverge at high
> learning rates (the network collapses to a constant output). Use `--lr 5e-4`
> for deep nets. The tower is initialized near-identity so it starts as a
> pass-through and learns gradually.

## 3. Quantize to integer weights

```bash
# always snapshot the checkpoint first, so a running trainer can't shift weights
# between quantizing and the parity check
cp compact_champion.pt /tmp/snap.pt
python quantize.py --model /tmp/snap.pt --out ../models/compact_champion.qbin
# deep nets:  python quantize_deep.py --model /tmp/snap.pt --out ../models/deep_b4.qbin
```

The quantizer picks the largest power-of-two scale that keeps every weight inside
the 16-bit range and writes that scale into the file header, so the C++ engine
stays bit-exact with no rebuild. Activations are clipped to a fixed range.

## 4. Verify parity (mandatory)

Build the engine and run the parity test. It re-evaluates a fixed set of
positions in C++ and compares against the Python integer reference:

```bash
cd ../engine && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/parity        # compact nets — must report 0 mismatches
./build/parity_deep   # deep nets
```

If parity fails, any strength measurement is meaningless — the C++ engine isn't
playing the network you trained. Parity gates everything.

## File guide

| file | role |
|---|---|
| `target.py` | the locked evaluation → win-probability conversion |
| `encode.py` | board → 768-feature sparse piece-square indices |
| `build_dataset.py` | labelled positions → training shards |
| `model.py` | the compact fully-connected network |
| `train.py` | trainer for compact networks |
| `deepmodel.py` | the deep residual-tower network |
| `train_deep.py` | trainer for deep networks (warmup + cosine + grad-clip) |
| `quantize.py` / `quantize_deep.py` | float → 16-bit integer `.qbin`, with a golden reference |
| `make_parity.py` / `make_parity_deep.py` | emit the fixtures the C++ parity tests check against |
| `arch_experiment.py` | the width × depth sweep that generated the scaling study |

## requirements.txt

```
torch>=2.6
python-chess>=1.11
numpy
pyarrow
```
