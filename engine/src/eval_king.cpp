// eval_king.cpp — king-relative deep judge: loader, from-scratch eval, and the
// incremental accumulator. From-scratch eval must match quantize_king.eval_int_king
// bit-for-bit (verified by parity_king); the incremental accumulator must match the
// from-scratch summary bit-for-bit (verified by a differential test in the engine).

#include "eval_king.h"
#include "board.h"
#include <cstdio>
#include <cstring>
#include <immintrin.h>

namespace chess {

static inline int hsum8k(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
static inline int dotk(const int16_t* a, const int16_t* b, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    int s = hsum8k(acc);
    for (; i < n; ++i) s += (int)a[i] * (int)b[i];
    return s;
}
static inline long rdivk(long num, long den) {
    return (num >= 0) ? (num + den / 2) / den : -((-num + den / 2) / den);
}

bool KingNet::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "eval_king: cannot open %s\n", path.c_str()); return false; }
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "CHESSKG1", 8) != 0) {
        fprintf(stderr, "eval_king: bad magic\n"); fclose(f); return false;
    }
    int32_t hdr[6];
    if (fread(hdr, sizeof(int32_t), 6, f) != 6) { fclose(f); return false; }
    in_size = hdr[0]; first_dim = hdr[1]; blocks = hdr[2]; hidden = hdr[3];
    weight_scale = hdr[4]; act_max = hdr[5];
    combined = 2 * first_dim;
    if (in_size != KING_FEATS) { fprintf(stderr, "eval_king: in_size != 40960\n"); fclose(f); return false; }

    auto rd = [&](std::vector<int16_t>& v, long n) -> bool {
        v.resize(n);
        return (long)fread(v.data(), sizeof(int16_t), n, f) == n;
    };
    std::vector<int16_t> first_raw;
    bool ok = rd(first_raw, (long)first_dim * KING_FEATS) && rd(first_b, first_dim);
    tower.resize(blocks);
    for (int i = 0; i < blocks && ok; ++i) {
        ok = ok && rd(tower[i].fc1_W, (long)hidden * combined) && rd(tower[i].fc1_b, hidden)
                && rd(tower[i].fc2_W, (long)combined * hidden) && rd(tower[i].fc2_b, combined);
    }
    ok = ok && rd(head_W, combined) && rd(head_b, 1);
    long pos = ftell(f); fseek(f, 0, SEEK_END); long end = ftell(f); fclose(f);
    if (!ok || pos != end) { fprintf(stderr, "eval_king: size mismatch (%ld vs %ld)\n", pos, end); return false; }

    // Transpose first layer to feature-major (KING_FEATS x first_dim) so each
    // feature's column is contiguous — the layout the accumulator add/subtracts.
    first_WT.assign((size_t)KING_FEATS * first_dim, 0);
    for (int j = 0; j < first_dim; ++j)
        for (int i = 0; i < KING_FEATS; ++i)
            first_WT[(size_t)i * first_dim + j] = first_raw[(size_t)j * KING_FEATS + i];
    return true;
}

// ---- king-relative feature indices for ONE perspective (mirrors king_encode.py) ----
// kind = piece_type(0..4 pawn..queen) + (0 own else 5); kings excluded as pieces.
//   feature = (king_sq*64 + piece_sq) * 10 + kind   (squares mirrored ^56 for Black)
void king_perspective_indices(const Board& b, Color persp, int* out, int& count) {
    count = 0;
    bool mirror = (persp == BLACK);
    int ksq = b.king_sq(persp);
    if (ksq < 0) return;
    int k = mirror ? (ksq ^ 56) : ksq;
    for (int sq = 0; sq < 64; ++sq) {
        int t = b.board[sq];
        if (t == NO_PIECE || t == KING) continue;          // skip empties and kings
        int c = b.color_on[sq];
        int s = mirror ? (sq ^ 56) : sq;
        int kind = t + (c == persp ? 0 : 5);               // PAWN..QUEEN are 0..4
        out[count++] = (k * 64 + s) * 10 + kind;
    }
}

// Compute one perspective's clipped summary from scratch into `out` (int16, 0..AM).
static void summary_from_scratch(const KingNet& net, const Board& b, Color persp,
                                 int16_t* out) {
    const int WS = net.weight_scale, AM = net.act_max, D = net.first_dim;
    int idx[32]; int n;
    king_perspective_indices(b, persp, idx, n);
    static thread_local std::vector<int32_t> acc;
    acc.assign(D, 0);
    for (int j = 0; j < D; ++j) acc[j] = net.first_b[j];
    for (int t = 0; t < n; ++t) {
        const int16_t* col = &net.first_WT[(size_t)idx[t] * D];
        int j = 0;
        for (; j + 8 <= D; j += 8) {
            __m256i a = _mm256_loadu_si256((const __m256i*)(acc.data() + j));
            __m128i w16 = _mm_loadu_si128((const __m128i*)(col + j));
            __m256i w = _mm256_cvtepi16_epi32(w16);
            _mm256_storeu_si256((__m256i*)(acc.data() + j), _mm256_add_epi32(a, w));
        }
        for (; j < D; ++j) acc[j] += col[j];
    }
    for (int j = 0; j < D; ++j) {
        long q = rdivk((long)acc[j] * AM, WS);
        out[j] = (int16_t)(q < 0 ? 0 : (q > AM ? AM : q));
    }
}

// Run the deep tower + head on a combined signal `s` (length combined, int16 0..AM).
static int tower_and_head(const KingNet& net, int16_t* s) {
    const int WS = net.weight_scale, AM = net.act_max, C = net.combined, H = net.hidden;
    const long S1 = (long)WS * AM;
    static thread_local std::vector<int16_t> h;
    for (const KingBlock& blk : net.tower) {
        h.assign(H, 0);
        for (int k = 0; k < H; ++k) {
            long a1 = (long)dotk(&blk.fc1_W[(size_t)k * C], s, C) + (long)blk.fc1_b[k] * AM;
            long q = rdivk(a1, WS);
            h[k] = (int16_t)(q < 0 ? 0 : (q > AM ? AM : q));
        }
        for (int j = 0; j < C; ++j) {
            long a2 = (long)dotk(&blk.fc2_W[(size_t)j * H], h.data(), H) + (long)blk.fc2_b[j] * AM;
            long d = rdivk(a2, WS);
            long v = (long)s[j] + d;
            s[j] = (int16_t)(v < 0 ? 0 : (v > AM ? AM : v));
        }
    }
    long acc2 = (long)dotk(net.head_W.data(), s, C) + (long)net.head_b[0] * AM;
    return (int)rdivk(acc2, S1);
}

// From-scratch eval: both perspectives fresh, then tower+head. Matches the golden.
int evaluate_king_scratch(const KingNet& net, const Board& b) {
    const int D = net.first_dim;
    static thread_local std::vector<int16_t> s;
    s.assign(net.combined, 0);
    Color stm = b.stm;
    summary_from_scratch(net, b, stm, s.data());          // side-to-move first
    summary_from_scratch(net, b, (Color)(stm ^ 1), s.data() + D);  // opponent second
    return tower_and_head(net, s.data());
}

// ---- Incremental accumulator ----------------------------------------------
// The accumulator holds the first layer's PRE-activation sum: acc.v[j] =
// first_b[j] + sum over active features f of first_W[f][j]. Clipping happens only
// when we read it for eval. Maintaining the pre-activation sum is what makes the
// add/subtract exact: clipping is not linear, so we must clip AFTER summing, never
// fold it into the running total.

void acc_refresh(const KingNet& net, const Board& b, Color persp, Accumulator& acc) {
    const int D = net.first_dim;
    acc.v.assign(D, 0);
    for (int j = 0; j < D; ++j) acc.v[j] = net.first_b[j];
    int idx[32]; int n;
    king_perspective_indices(b, persp, idx, n);
    for (int t = 0; t < n; ++t) {
        const int16_t* col = &net.first_WT[(size_t)idx[t] * D];
        int j = 0;
        for (; j + 8 <= D; j += 8) {
            __m256i a = _mm256_loadu_si256((const __m256i*)(acc.v.data() + j));
            __m128i w16 = _mm_loadu_si128((const __m128i*)(col + j));
            __m256i w = _mm256_cvtepi16_epi32(w16);
            _mm256_storeu_si256((__m256i*)(acc.v.data() + j), _mm256_add_epi32(a, w));
        }
        for (; j < D; ++j) acc.v[j] += col[j];
    }
    acc.computed = true;
}

void acc_apply_feature(const KingNet& net, Accumulator& acc, int feature, int sign) {
    const int D = net.first_dim;
    const int16_t* col = &net.first_WT[(size_t)feature * D];
    if (sign > 0) {
        int j = 0;
        for (; j + 8 <= D; j += 8) {
            __m256i a = _mm256_loadu_si256((const __m256i*)(acc.v.data() + j));
            __m128i w16 = _mm_loadu_si128((const __m128i*)(col + j));
            __m256i w = _mm256_cvtepi16_epi32(w16);
            _mm256_storeu_si256((__m256i*)(acc.v.data() + j), _mm256_add_epi32(a, w));
        }
        for (; j < D; ++j) acc.v[j] += col[j];
    } else {
        int j = 0;
        for (; j + 8 <= D; j += 8) {
            __m256i a = _mm256_loadu_si256((const __m256i*)(acc.v.data() + j));
            __m128i w16 = _mm_loadu_si128((const __m128i*)(col + j));
            __m256i w = _mm256_cvtepi16_epi32(w16);
            _mm256_storeu_si256((__m256i*)(acc.v.data() + j), _mm256_sub_epi32(a, w));
        }
        for (; j < D; ++j) acc.v[j] -= col[j];
    }
}

// Clip an accumulator's pre-activation sums into int16 0..AM (the summary the
// tower reads). Mirrors the scale/clip in summary_from_scratch exactly.
static void clip_acc(const KingNet& net, const Accumulator& acc, int16_t* out) {
    const int WS = net.weight_scale, AM = net.act_max, D = net.first_dim;
    for (int j = 0; j < D; ++j) {
        long q = rdivk((long)acc.v[j] * AM, WS);
        out[j] = (int16_t)(q < 0 ? 0 : (q > AM ? AM : q));
    }
}

int evaluate_king_from_acc(const KingNet& net, Color stm,
                           const Accumulator& white_acc, const Accumulator& black_acc) {
    const int D = net.first_dim;
    static thread_local std::vector<int16_t> s;
    s.assign(net.combined, 0);
    // Side-to-move perspective first, opponent second — same ordering as scratch.
    const Accumulator& stm_acc = (stm == WHITE) ? white_acc : black_acc;
    const Accumulator& opp_acc = (stm == WHITE) ? black_acc : white_acc;
    clip_acc(net, stm_acc, s.data());
    clip_acc(net, opp_acc, s.data() + D);
    return tower_and_head(net, s.data());
}

}  // namespace chess
