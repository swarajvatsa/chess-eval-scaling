// eval_deep.cpp — integer DEEP residual judge (must match quantize_deep.py::eval_int_deep).

#include "eval_deep.h"
#include "eval.h"      // for active_indices (shared with the shallow eval)
#include <cstdio>
#include <cstring>
#include <immintrin.h>

namespace chess {

// --- AVX2 helpers (local copies; the shallow eval.cpp keeps its own static ones) ---
static inline int hsum8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
// Dot product of two int16 arrays length n, accumulated in int32 lanes.
static inline int doti16(const int16_t* a, const int16_t* b, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    int s = hsum8(acc);
    for (; i < n; ++i) s += (int)a[i] * (int)b[i];
    return s;
}

// Integer round-half-away-from-zero — identical to quantize_deep._round_div and
// the shallow engine's divide, so all three agree to the integer.
static inline long rdiv(long num, long den) {
    return (num >= 0) ? (num + den / 2) / den : -((-num + den / 2) / den);
}

bool DeepNet::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "eval_deep: cannot open %s\n", path.c_str()); return false; }
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "CHESSDP1", 8) != 0) {
        fprintf(stderr, "eval_deep: bad magic in %s\n", path.c_str()); fclose(f); return false;
    }
    int32_t hdr[6];
    if (fread(hdr, sizeof(int32_t), 6, f) != 6) { fclose(f); return false; }
    in_size = hdr[0]; dim = hdr[1]; blocks = hdr[2]; hidden = hdr[3];
    weight_scale = hdr[4]; act_max = hdr[5];
    if (in_size != 768) { fprintf(stderr, "eval_deep: in_size != 768\n"); fclose(f); return false; }

    auto rd = [&](std::vector<int16_t>& v, int n) -> bool {
        v.resize(n);
        return (int)fread(v.data(), sizeof(int16_t), n, f) == n;
    };

    // First layer is stored output-major (dim x in_size); read then transpose to
    // input-major (in_size x dim) for the sparse add.
    std::vector<int16_t> first_raw;
    bool ok = rd(first_raw, dim * in_size) && rd(first_b, dim);
    tower.resize(blocks);
    for (int i = 0; i < blocks && ok; ++i) {
        ok = ok && rd(tower[i].fc1_W, hidden * dim) && rd(tower[i].fc1_b, hidden)
                && rd(tower[i].fc2_W, dim * hidden) && rd(tower[i].fc2_b, dim);
    }
    ok = ok && rd(head_W, dim) && rd(head_b, 1);

    long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fclose(f);
    if (!ok || pos != end) {
        fprintf(stderr, "eval_deep: size mismatch (%ld vs %ld)\n", pos, end); return false;
    }

    first_WT.assign((size_t)in_size * dim, 0);
    for (int j = 0; j < dim; ++j)
        for (int i = 0; i < in_size; ++i)
            first_WT[(size_t)i * dim + j] = first_raw[(size_t)j * in_size + i];
    return true;
}

// Forward pass — bit-identical arithmetic to eval_int_deep.
//
//   first:  s[j] = clip(round((first_b[j] + sum_active first_W[j,i]) * AM / WS))
//   block:  h[k] = clip(round((fc1_b[k]*AM + sum_j fc1_W[k,j]*s[j]) / WS))   [*AM/(WS*AM)=/WS]
//           d[j] = round((fc2_b[j]*AM + sum_k fc2_W[j,k]*h[k]) / WS)
//           s[j] = clip(s[j] + d[j])
//   head:   raw  = round((head_b*AM + sum_j head_W[j]*s[j]) / (WS*AM))
int evaluate_deep(const DeepNet& net, const Board& b) {
    int idx[40]; int n;
    active_indices(b, idx, n);
    const int WS = net.weight_scale, AM = net.act_max, D = net.dim, H = net.hidden;
    const long S1 = (long)WS * AM;

    // ---- first layer (sparse): accumulate active input rows of the transposed matrix
    static thread_local std::vector<int32_t> acc;
    acc.assign(D, 0);
    for (int j = 0; j < D; ++j) acc[j] = net.first_b[j];
    for (int t = 0; t < n; ++t) {
        const int16_t* row = &net.first_WT[(size_t)idx[t] * D];
        int j = 0;
        for (; j + 8 <= D; j += 8) {
            __m256i a = _mm256_loadu_si256((const __m256i*)(acc.data() + j));
            __m128i w16 = _mm_loadu_si128((const __m128i*)(row + j));
            __m256i w = _mm256_cvtepi16_epi32(w16);
            _mm256_storeu_si256((__m256i*)(acc.data() + j), _mm256_add_epi32(a, w));
        }
        for (; j < D; ++j) acc[j] += row[j];
    }
    // The running signal `s`, kept as int16 in 0..AM so the block dots can use madd.
    static thread_local std::vector<int16_t> s;
    s.assign(D, 0);
    for (int j = 0; j < D; ++j) {
        long q = rdiv((long)acc[j] * AM, WS);
        s[j] = (int16_t)(q < 0 ? 0 : (q > AM ? AM : q));
    }

    // ---- residual blocks ----
    static thread_local std::vector<int16_t> h;
    for (const DeepBlock& blk : net.tower) {
        h.assign(H, 0);
        // fc1 + clip -> h
        for (int k = 0; k < H; ++k) {
            long a1 = (long)doti16(&blk.fc1_W[(size_t)k * D], s.data(), D)
                    + (long)blk.fc1_b[k] * AM;
            long q = rdiv(a1, WS);            // *AM/(WS*AM) == /WS
            h[k] = (int16_t)(q < 0 ? 0 : (q > AM ? AM : q));
        }
        // fc2 -> delta, then residual skip + re-clip (in place on s)
        for (int j = 0; j < D; ++j) {
            long a2 = (long)doti16(&blk.fc2_W[(size_t)j * H], h.data(), H)
                    + (long)blk.fc2_b[j] * AM;
            long d = rdiv(a2, WS);
            long v = (long)s[j] + d;
            s[j] = (int16_t)(v < 0 ? 0 : (v > AM ? AM : v));
        }
    }

    // ---- head ----
    long acc2 = (long)doti16(net.head_W.data(), s.data(), D) + (long)net.head_b[0] * AM;
    return (int)rdiv(acc2, S1);
}

}  // namespace chess
