// eval_deep.h — The integer DEEP residual judge for the C++ engine.
//
// This is the C++ twin of src/quantize_deep.py::eval_int_deep. It loads the deep
// judge's quantized weights (magic "CHESSDP1") and runs the exact same integer
// arithmetic — a sparse first layer, then N residual blocks, then a head — so the
// engine's score equals the Python golden score to the integer. We verify that
// with a parity test (parity_deep) before trusting it in a game.
//
// The deep judge is the architecture we proved strongest: depth + residual skip
// connections let the eval chain multi-step reasoning. Each block is two dense
// layers, a clipped activation, an integer skip-add, and a re-clip — every piece
// is something the engine already does, so the integer port is low-risk.

#pragma once
#include "board.h"
#include <cstdint>
#include <string>
#include <vector>

namespace chess {

// One residual block's quantized weights (int16, weight-scale fixed-point).
struct DeepBlock {
    std::vector<int16_t> fc1_W;  // (hidden x dim)
    std::vector<int16_t> fc1_b;  // (hidden)
    std::vector<int16_t> fc2_W;  // (dim x hidden)
    std::vector<int16_t> fc2_b;  // (dim)
};

struct DeepNet {
    int in_size = 0, dim = 0, blocks = 0, hidden = 0;
    int weight_scale = 0, act_max = 0;

    // First layer stored TRANSPOSED to input-major (in_size x dim), so the sparse
    // "sum the active inputs' rows" is a sequence of contiguous int16 adds — the
    // same cache-friendly trick the shallow eval uses, and the layout the future
    // incremental accumulator will update in place.
    std::vector<int16_t> first_WT;  // (in_size x dim) transposed
    std::vector<int16_t> first_b;   // (dim)
    std::vector<DeepBlock> tower;   // N blocks
    std::vector<int16_t> head_W;    // (1 x dim)
    std::vector<int16_t> head_b;    // (1)

    bool load(const std::string& path);
};

// Active 768-input indices for the side to move (identical to encode.py / the
// shallow eval's active_indices). Declared in eval.h already; we reuse it.

// Run the integer deep judge; returns the raw centipawn-like score from the side
// to move's point of view — equal to eval_int_deep's integer output.
int evaluate_deep(const DeepNet& net, const Board& b);

}  // namespace chess
