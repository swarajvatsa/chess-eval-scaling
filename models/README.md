# Models

Five ready-to-play position evaluators ("judges"), shipped as quantized 16-bit
integer weights (`.qbin`). Each file begins with an 8-byte magic tag the engine
uses to auto-detect the architecture, so you never have to specify it — just
point the engine's `EvalFile` option at the file.

All five were trained on the same ~289M-position dataset with the same
win-probability objective for 30 epochs, then quantized. The C++ engine evaluates
with these integer weights bit-for-bit identically to the Python originals
(verified by `engine/tests/parity`).

| file | architecture | shape | params | response time (µs/eval) | validation loss |
|---|---|---|---:|---:|---:|
| `compact_tiny.qbin` | compact fully-connected | 768→128→32→1 | 102,593 | 0.77 | 0.03100 |
| `compact_small.qbin` | compact fully-connected | 768→512→64→1 | 426,625 | 3.31 | 0.02555 |
| `compact_champion.qbin` | compact fully-connected | 768→1024→256→1 | 1,050,113 | 16.05 | 0.02252 |
| `compact_wide.qbin` | compact fully-connected | 768→2048→256→1 | 2,099,713 | 27.58 | 0.02122 |
| `deep_b4.qbin` | deep residual tower | 768→512→4×block(512)→1 | 2,495,489 | 107.0 | 0.02032 |

- **architecture** — *compact* nets are wide shallow fully-connected networks;
  *deep* nets are residual-block towers. Inputs are a sparse piece-square
  encoding (768 features: 6 piece types × 2 colours × 64 squares), seen from the
  side-to-move's perspective.
- **response time** — microseconds to evaluate one position in the compiled C++
  engine, measured with `engine/build/evalspeed`. This is the cost that competes
  with thinking time: faster networks search more positions per second.
- **validation loss** — mean-squared error in win-probability space on held-out
  positions. Lower means a better judge of a single position. Note it *improves*
  monotonically with size while playing strength does not — see
  [../ANALYSIS.md](../ANALYSIS.md).

## Which one should I use?

- **`compact_tiny`** — the fastest, and the default play network. At ordinary
  thinking times its speed lets it out-search the bigger nets.
- **`compact_champion`** — a strong, robust all-rounder if you want a single
  larger network.
- **`deep_b4`** — the best *judge* (lowest loss) but the slowest; it only pays off
  at long time controls (tens of seconds per move), where there's time to search
  deeply enough to use its better judgment.

## Regenerating these files

Each `.qbin` is produced from a trained PyTorch checkpoint by the quantizer:

```bash
# compact nets
python training/quantize.py --model compact_tiny.pt --out models/compact_tiny.qbin
# deep nets
python training/quantize_deep.py --model deep_b4.pt --out models/deep_b4.qbin
```

Always build the engine and run `./parity` afterward — it must report **0
mismatches**, confirming the integer weights reproduce the float network exactly.
