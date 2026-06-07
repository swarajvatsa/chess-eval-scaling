#!/usr/bin/env bash
# Build the engine and its test binaries (Release, native AVX2).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO/engine"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
echo "built: $REPO/engine/build/{engine,perft,parity,parity_deep,evalspeed,searchbench}"
