# chess-eval-scaling

A from-scratch chess engine and an empirical study of one question:
**how big should a chess engine's "brain" be?**

The engine's position evaluator is a small neural network — a "judge" — trained
in Python and shipped as compact integer weights that a C++ alpha-beta search
loads and plays with. We trained a family of judges from ~100K to ~2.5M
parameters, all on the same data, and measured how each one actually plays across
a range of thinking times and CPU cores.

## The headline result

A bigger network is a strictly **better judge** of any single position — but
that does **not** make a better engine on its own. Whether judgment or raw speed
wins depends on the clock:

- **Short on time:** the small, fast network wins — it's cheap to evaluate, so it
  searches far more positions.
- **Plenty of time:** the bigger network's better judgment catches up, because it
  gains more strength per extra second of thinking.
- **The real lever is search** — but you can't brute-force it: more compute buys
  linearly more positions examined, yet only logarithmically more lookahead.

![Strength vs thinking time](docs/images/elo_vs_timecontrol.png)

The full write-up, with every measured number and confidence range, is in
**[ANALYSIS.md](ANALYSIS.md)**.

## What's in here

```
chess-eval-scaling/
├── ANALYSIS.md          the findings, with plots and data tables
├── engine/              the C++ playing engine (build this to play)
│   ├── src/             board, move generation, search, the integer evaluator
│   └── tests/           correctness checks (move-gen, evaluator parity, speed)
├── training/            the Python pipeline that produces the networks
├── models/              5 ready-to-play networks (quantized integer weights)
├── analysis/            the data (CSV) and plotting code behind every figure
└── scripts/             build / play helpers
```

## Quick start — build and play

The engine is standalone C++17 (needs CMake and an AVX2-capable CPU). No
libraries required to play.

```bash
# build
cd engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# verify correctness (move generation + evaluator bit-exactness)
cd build
./perft          # move generation is exact
./parity         # C++ integer evaluator == Python reference, 0 mismatches

# play with the default config: the fast network, 8 search threads
#   (capped to your machine's core count)
../../scripts/play.sh
```

The engine speaks the standard UCI protocol, so you can also point any UCI chess
GUI at `engine/build/engine` and choose a network with the `EvalFile` option:

```
setoption name EvalFile value models/compact_champion.qbin
setoption name Threads value 8
setoption name Hash value 64
position startpos
go movetime 1000
```

## The shipped networks

Five ready-to-play networks live in `models/` (details in
[models/README.md](models/README.md)):

| file | params | response time | best for |
|---|---:|---:|---|
| `compact_tiny.qbin` | 102K | 0.77 µs/eval | fastest; the default play network |
| `compact_small.qbin` | 427K | 3.3 µs/eval | balanced |
| `compact_champion.qbin` | 1.05M | 16 µs/eval | strong all-rounder |
| `compact_wide.qbin` | 2.10M | 28 µs/eval | widest compact net |
| `deep_b4.qbin` | 2.50M | 107 µs/eval | best judge, needs long time controls |

## Train your own

```bash
pip install -r training/requirements.txt

# 1. build training shards from a folder of labelled positions
python training/build_dataset.py --parquet data/positions --out shards/

# 2. train a compact network (or use train_deep.py for a residual-tower net)
python training/train.py --shards shards --out my_net.pt --epochs 30

# 3. quantize to integer weights the C++ engine can load,
#    then build the engine and run ./parity — it must report 0 mismatches
cp my_net.pt /tmp/snap.pt
python training/quantize.py --model /tmp/snap.pt --out models/my_net.qbin
```

See [training/README.md](training/README.md) for the full pipeline, the
win-probability objective, and the integer-quantization scheme.

## How it works, briefly

- **Train in Python, play in C++.** The network is trained with PyTorch, then its
  weights are converted to 16-bit integers and written to a `.qbin` file. The C++
  engine reads that file and evaluates positions with hand-vectorized integer
  arithmetic that reproduces the Python network **bit-for-bit** (verified by the
  `parity` tests).
- **The search** is alpha-beta with principal-variation search, a quiescence
  search, a lock-free transposition table, and multi-threaded parallel search.
  It uses the network only to score leaf positions.
- **The objective** is win probability: each training position's known evaluation
  is converted to a probability of winning, and the network learns to predict it.

## License

MIT — see [LICENSE](LICENSE).
