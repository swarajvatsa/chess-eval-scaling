// eval.h — The integer brain: load the quantized weights and score a position.
//
// This mirrors, in C++, exactly what src/quantize.py's eval_int does in Python.
// The two MUST agree to the integer — that bit-for-bit parity is how we know the
// C++ port didn't subtly corrupt the brain. We verify it with a parity test
// before trusting any game the engine plays.

#pragma once
#include "board.h"
#include <cstdint>
#include <string>

namespace chess {

// Architecture constants — must match the trained net and the .qbin header.
constexpr int IN_SIZE = 768;     // 6 piece-types x 2 colors x 64 squares
// hidden / hidden2 are read from the file header at load time (1024 / 256 here).

struct Net {
    int in_size = 0, hidden = 0, hidden2 = 0;
    int weight_scale = 0;        // fixed-point scale of the weights (e.g. 256)
    int act_max = 0;             // clipped-relu integer ceiling (e.g. 255)
    // Weights kept as int16 so AVX2 can load and multiply them directly. W0 is
    // stored TRANSPOSED to "input-major" layout (W0T[input][hidden]) so that
    // summing the active inputs' rows is a sequence of contiguous int16 adds —
    // cache-friendly for the sparse first layer. W1/W2 stay output-major (each
    // output's weight row is contiguous) for SIMD dot products against the
    // activation vector.
    std::vector<int16_t> W0T;    // (in_size x hidden)  -- transposed
    std::vector<int16_t> b0;     // (hidden)
    std::vector<int16_t> W1, b1; // (hidden2 x hidden), (hidden2)
    std::vector<int16_t> W2, b2; // (1 x hidden2), (1)

    // Load the quantized weight binary written by quantize.py. Returns false on
    // any header/size mismatch (so a wrong or truncated file is caught, not run).
    bool load(const std::string& path);
};

// Fill `out` with the active 768-input indices for the side to move, mirroring
// encode.py's active_indices (mover's point of view, board flipped for Black).
void active_indices(const Board& b, int* out, int& count);

// Run the integer brain on a position; returns the raw centipawn-like score from
// the side-to-move's point of view (the same number quantize.eval_int returns).
int evaluate(const Net& net, const Board& b);

}  // namespace chess
