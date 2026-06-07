// eval_king.h — The king-relative deep judge with an INCREMENTAL accumulator.
//
// This is the 100x path. The first layer maps 40960 king-relative features per
// perspective into a `first_dim`-wide summary. Computing that summary from scratch
// is ~30 features x first_dim adds per perspective per position — but because the
// king-relative features barely change when a non-king piece moves, we maintain
// the summary INCREMENTALLY: on each move we add the few feature-columns that
// turned on and subtract the few that turned off, instead of recomputing. Only the
// cheap deep tower (on the small 2*first_dim signal) runs in full every eval.
//
// Binary format magic "CHESSKG1":
//   in_size(=40960) first_dim blocks hidden weight_scale act_max   (6 x int32)
//   first_W (first_dim x 40960)   first_b (first_dim)
//   for each block: fc1_W (hidden x 2dim) fc1_b (hidden) fc2_W (2dim x hidden) fc2_b (2dim)
//   head_W (1 x 2dim)  head_b (1)
//
// The first layer is stored TRANSPOSED to feature-major (40960 x first_dim) so each
// feature's column is contiguous — the layout the accumulator adds/subtracts.

#pragma once
#include "board.h"
#include <cstdint>
#include <string>
#include <vector>

namespace chess {

constexpr int KING_FEATS = 40960;   // 64 king_sq x 64 piece_sq x 10 kinds

struct KingBlock {
    std::vector<int16_t> fc1_W, fc1_b, fc2_W, fc2_b;
};

struct KingNet {
    int in_size = 0, first_dim = 0, blocks = 0, hidden = 0;
    int weight_scale = 0, act_max = 0;
    int combined = 0;                 // 2 * first_dim

    std::vector<int16_t> first_WT;    // (KING_FEATS x first_dim) feature-major
    std::vector<int16_t> first_b;     // (first_dim)
    std::vector<KingBlock> tower;
    std::vector<int16_t> head_W;      // (1 x combined)
    std::vector<int16_t> head_b;      // (1)

    bool load(const std::string& path);
};

// An accumulator: the first layer's pre-activation sum for ONE perspective. Kept
// as int32 because the running sum (bias + up to ~30 weight columns) needs the
// width. There are two per board (one per color).
struct Accumulator {
    std::vector<int32_t> v;           // (first_dim) pre-activation sums
    bool computed = false;            // false => must refresh from scratch
};

// Active king-relative feature indices for ONE perspective (mirrors king_encode.py).
void king_perspective_indices(const Board& b, Color persp, int* out, int& count);

// From-scratch eval (both perspectives recomputed). Matches quantize_king's golden
// eval_int_king bit-for-bit. Used by the parity test and as the correctness oracle
// for the incremental accumulator.
int evaluate_king_scratch(const KingNet& net, const Board& b);

// ---- Incremental accumulator API (the 100x) -------------------------------
// Refresh one perspective's accumulator from scratch (bias + active columns) — the
// O(30) full rebuild, used at init and after the perspective's own king moves.
void acc_refresh(const KingNet& net, const Board& b, Color persp, Accumulator& acc);

// Apply ONE feature column to an accumulator: sign=+1 adds (feature turned on),
// sign=-1 subtracts (feature turned off). O(first_dim).
void acc_apply_feature(const KingNet& net, Accumulator& acc, int feature, int sign);

// Evaluate directly from two already-maintained accumulators (clip -> tower -> head),
// without recomputing the first layer. Must equal evaluate_king_scratch.
int evaluate_king_from_acc(const KingNet& net, Color stm,
                           const Accumulator& white_acc, const Accumulator& black_acc);

}  // namespace chess
