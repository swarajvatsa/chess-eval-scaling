#!/usr/bin/env bash
# Launch the engine with the default config: the fast network at up to 8 search
# threads (never more than the machine's physical cores). Pass a different .qbin
# as the first argument to play a different network.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENG="$REPO/engine/build/engine"
NET="${1:-$REPO/models/compact_tiny.qbin}"
CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
THREADS=$(( 8 < CORES ? 8 : CORES ))
if [ ! -x "$ENG" ]; then
  echo "engine not built yet — run scripts/build.sh first" >&2; exit 1
fi
echo "engine: $ENG"
echo "network: $NET"
echo "threads: $THREADS (machine has $CORES cores)"
echo "--- type UCI commands, e.g.:  position startpos / go movetime 1000 / quit ---"
printf 'setoption name EvalFile value %s\nsetoption name Threads value %d\nsetoption name Hash value 64\n' \
  "$NET" "$THREADS" | cat - /dev/stdin | "$ENG"
