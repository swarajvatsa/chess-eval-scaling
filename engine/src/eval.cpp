// eval.cpp — integer brain implementation (must match quantize.py::eval_int).

#include "eval.h"
#include <cstdio>
#include <cstring>
#include <immintrin.h>   // AVX2 intrinsics for the fast matrix multiplies

namespace chess {

// Horizontal sum of an 8-lane int32 AVX2 vector -> a single int.
static inline int hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// Dot product of two int16 arrays of length n, accumulated in int32. Uses AVX2
// _mm256_madd_epi16 (multiplies 16 int16 pairs and adds adjacent results into 8
// int32 lanes) — the workhorse that makes the hidden-layer matmul fast.
static inline int dot_i16(const int16_t* a, const int16_t* b, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    int s = hsum_epi32(acc);
    for (; i < n; ++i) s += (int)a[i] * (int)b[i];   // tail
    return s;
}

// Piece-type -> channel index 0..5. MUST match encode.py's _PIECE_TO_CHANNEL,
// which happens to be our enum order already (PAWN=0..KING=5).
// active_indices: build the list of "on" 768-switches for the side to move.
//
// This is the C++ twin of encode.py. The key subtlety, copied exactly: we encode
// from the MOVER's point of view. When Black is to move we vertically mirror each
// square (sq ^ 56 flips the rank: a1<->a8) so the mover always "plays up", and we
// put the mover's own pieces in channels 0..5 and the opponent's in 6..11.
//
//   switch index = channel * 64 + (possibly-mirrored square)
//   channel      = piece_type + (0 if mover's piece else 6)
void active_indices(const Board& b, int* out, int& count) {
    count = 0;
    Color us = b.stm;
    for (int sq = 0; sq < 64; ++sq) {
        int t = b.board[sq];
        if (t == NO_PIECE) continue;
        int c = b.color_on[sq];
        // Mirror the square when Black is to move (sq ^ 56 flips rank, same as
        // python-chess square_mirror). When White to move, leave it.
        int esq = (us == WHITE) ? sq : (sq ^ 56);
        int color_block = (c == us) ? 0 : 6;
        int channel = t + color_block;
        out[count++] = channel * 64 + esq;
    }
}

bool Net::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "eval: cannot open %s\n", path.c_str()); return false; }
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "CHESSBR1", 8) != 0) {
        fprintf(stderr, "eval: bad magic in %s\n", path.c_str()); fclose(f); return false;
    }
    int32_t hdr[5];
    if (fread(hdr, sizeof(int32_t), 5, f) != 5) { fclose(f); return false; }
    in_size = hdr[0]; hidden = hdr[1]; hidden2 = hdr[2];
    weight_scale = hdr[3]; act_max = hdr[4];
    if (in_size != IN_SIZE) { fprintf(stderr, "eval: in_size mismatch\n"); fclose(f); return false; }

    auto read_block = [&](std::vector<int16_t>& v, int n) -> bool {
        v.resize(n);
        return (int)fread(v.data(), sizeof(int16_t), n, f) == n;
    };
    // The file stores W0 output-major as (hidden x in_size). We read it into a
    // temporary, then TRANSPOSE into W0T (in_size x hidden) so the first layer's
    // sparse "sum the active inputs' rows" is contiguous and cache-friendly.
    std::vector<int16_t> W0_raw;
    bool ok = read_block(W0_raw, hidden * in_size) && read_block(b0, hidden)
           && read_block(W1, hidden2 * hidden) && read_block(b1, hidden2)
           && read_block(W2, 1 * hidden2)       && read_block(b2, 1);
    long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fclose(f);
    if (!ok || pos != end) { fprintf(stderr, "eval: size mismatch (%ld vs %ld)\n", pos, end); return false; }

    W0T.assign((size_t)in_size * hidden, 0);
    for (int j = 0; j < hidden; ++j)
        for (int i = 0; i < in_size; ++i)
            W0T[(size_t)i * hidden + j] = W0_raw[(size_t)j * in_size + i];
    return true;
}

// Forward pass in integers — identical arithmetic to quantize.py::eval_int.
//
//   Layer 0: inputs are 0/1, so acc0[j] = b0[j] + sum of W0[j, active_i].
//            real0 = acc0 / weight_scale; clipped-relu -> a0 in 0..act_max,
//            where a0 = clip(round(real0 * act_max), 0, act_max).
//   Layer 1: acc1[k] = sum_j W1[k,j]*a0[j] + b1[k]*act_max
//            (b1 lifted to the accumulator scale weight_scale*act_max).
//            real1 = acc1 / (weight_scale*act_max); a1 = clip(round(real1*act_max)).
//   Layer 2: acc2 = sum_k W2[k]*a1[k] + b2*act_max; raw = acc2 / (weight_scale*act_max).
//
// We round-half-away-from-zero with the (x>=0 ? +0.5 : -0.5) idiom on the divide,
// matching numpy.round's behavior closely enough that the integer results agree.
int evaluate(const Net& net, const Board& b) {
    int idx[40];
    int n;
    active_indices(b, idx, n);
    const int WS = net.weight_scale, AM = net.act_max, H = net.hidden, H2 = net.hidden2;
    const long S1 = (long)WS * AM;

    // ---- Layer 0 (sparse): a0[j] = clip(round((b0[j] + sum_active W0[j,i]) * AM / WS)) ----
    // Because inputs are 0/1, we just ADD the rows of the transposed weight matrix
    // for each active input. W0T is input-major so each row W0T[i*H .. i*H+H) is
    // contiguous — we add H int16 values per active input. We accumulate in int32.
    static thread_local std::vector<int32_t> acc0;
    acc0.assign(H, 0);
    for (int j = 0; j < H; ++j) acc0[j] = net.b0[j];
    for (int t = 0; t < n; ++t) {
        const int16_t* row = &net.W0T[(size_t)idx[t] * H];
        // Vectorized int16->int32 add of `row` into acc0.
        int j = 0;
        for (; j + 8 <= H; j += 8) {
            __m256i a = _mm256_loadu_si256((const __m256i*)(acc0.data() + j));
            // widen 8 int16 to 8 int32
            __m128i w16 = _mm_loadu_si128((const __m128i*)(row + j));
            __m256i w = _mm256_cvtepi16_epi32(w16);
            _mm256_storeu_si256((__m256i*)(acc0.data() + j), _mm256_add_epi32(a, w));
        }
        for (; j < H; ++j) acc0[j] += row[j];
    }
    // Activation: scale by AM, divide by WS (round-to-nearest), clip to [0, AM].
    // Store as int16 so the layer-1 SIMD dot can use _mm256_madd_epi16.
    static thread_local std::vector<int16_t> a0;
    a0.assign(H, 0);
    for (int j = 0; j < H; ++j) {
        long num = (long)acc0[j] * AM;
        long q = (num >= 0) ? (num + WS / 2) / WS : -((-num + WS / 2) / WS);
        a0[j] = (int16_t)(q < 0 ? 0 : (q > AM ? AM : q));
    }

    // ---- Layer 1 (dense): SIMD dot of each output row with a0 ----
    static thread_local std::vector<int16_t> a1;
    a1.assign(H2, 0);
    for (int k = 0; k < H2; ++k) {
        long acc = (long)dot_i16(&net.W1[(size_t)k * H], a0.data(), H);
        acc += (long)net.b1[k] * AM;
        long num = acc * (long)AM;
        long q = (num >= 0) ? (num + S1 / 2) / S1 : -((-num + S1 / 2) / S1);
        a1[k] = (int16_t)(q < 0 ? 0 : (q > AM ? AM : q));
    }

    // ---- Layer 2 (single output, no activation) ----
    long acc = (long)dot_i16(net.W2.data(), a1.data(), H2) + (long)net.b2[0] * AM;
    long q = (acc >= 0) ? (acc + S1 / 2) / S1 : -((-acc + S1 / 2) / S1);
    return (int)q;
}

}  // namespace chess
