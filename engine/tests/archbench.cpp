// archbench.cpp — How long does ONE neural-net forward pass ("one response") take,
// as a function of WIDTH, DEPTH, and total SIZE?
//
// We do NOT care what the net computes — only how fast. So we hardcode the
// ARCHITECTURE here in C++, fill it with random weights, and time the forward pass.
//
// Every layer is followed by a SIGMOID activation, exactly like a real net. This is
// the whole point: sigmoid is non-linear, so the layers CANNOT be folded into one
// matrix. D hidden layers means D genuine matrix multiplies that must run in
// sequence, each waiting on the previous one's activation. That is the cost the
// activations force us to pay, and it is what makes a "deep" net expensive.
//
// The forward pass is written the most optimized way we ship:
//   - hand-vectorized AVX2 fused-multiply-add (_mm256_fmadd_ps): 8 floats per insn,
//   - compiled -O3 -march=native (Zen3 AVX2+FMA),
//   - denormals flushed to zero (FTZ/DAZ) — a classic micro-benchmark trap where
//     tiny subnormal floats silently run ~100x slower,
//   - random weights are runtime data, so nothing constant-folds; each iteration
//     perturbs the input and feeds the previous output back, so the timing loop
//     cannot be hoisted or collapsed by the compiler.
//
// Net shape:   768 --(dense)--> W --(dense)--> W ... [D hidden layers] ... --> W --> 1
// Sweep:       width W  x  depth D.
// Report:      params (size), microseconds per forward pass, evals/sec, and the
//              effective GFLOP/s (vs ~57 GFLOP/s AVX2-FMA single-core peak @3.6GHz).

#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <vector>
#include <random>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>

using std::vector;

// Horizontal sum of an 8-lane AVX vector into one float.
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s  = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, s);
    s  = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

struct Layer {
    int in, out;
    vector<float> W;   // out*in, row-major
    vector<float> b;   // out
};

struct Net {
    vector<Layer> layers;
    mutable vector<float> buf_a, buf_b;
    long long params() const {
        long long p = 0; for (auto& L : layers) p += (long long)L.in * L.out + L.out; return p;
    }
    long long macs() const {   // multiply-accumulate ops in one forward pass
        long long m = 0; for (auto& L : layers) m += (long long)L.in * L.out; return m;
    }
};

static void build_net(Net& net, int input_dim, int W, int D, uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto mk = [&](int in, int out) {
        Layer L; L.in = in; L.out = out;
        L.W.resize((size_t)in * out); L.b.resize(out);
        // Xavier-ish scale keeps pre-activations ~O(1): no NaNs, no denormal storms.
        float scale = 1.0f / std::sqrt((float)in);
        std::uniform_real_distribution<float> d(-scale, scale);
        for (auto& w : L.W) w = d(rng);
        for (auto& x : L.b) x = d(rng);
        return L;
    };
    net.layers.clear();
    net.layers.push_back(mk(input_dim, W));         // input projection 768 -> W
    for (int i = 0; i < D; i++) net.layers.push_back(mk(W, W));  // deep tower W -> W
    net.layers.push_back(mk(W, 1));                 // output head W -> 1
    int maxdim = std::max(input_dim, W);
    net.buf_a.assign(maxdim, 0.f);
    net.buf_b.assign(maxdim, 0.f);
}

// y[o] = sigmoid( bias[o] + sum_i W[o,i]*x[i] ), vectorized over i with AVX2 FMA.
static inline void dense_sigmoid(const Layer& L, const float* x, float* y) {
    for (int o = 0; o < L.out; o++) {
        const float* w = &L.W[(size_t)o * L.in];
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= L.in; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), acc);
        float s = hsum256(acc);
        for (; i < L.in; i++) s += w[i] * x[i];     // scalar tail
        s += L.b[o];
        y[o] = 1.0f / (1.0f + expf(-s));            // sigmoid
    }
}

static inline float forward(const Net& net, const float* in_buf) {
    const float* x = in_buf;
    float* a = net.buf_a.data();
    float* b = net.buf_b.data();
    float* cur = a;
    for (size_t i = 0; i < net.layers.size(); i++) {
        dense_sigmoid(net.layers[i], x, cur);
        x = cur;
        cur = (cur == a) ? b : a;
    }
    return x[0];   // output head has out == 1
}

int main(int argc, char** argv) {
    // Flush denormals to zero — otherwise subnormal activations can silently tank
    // the timing by ~100x and corrupt the measurement.
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    int input_dim  = 768;
    double budget  = (argc > 1) ? atof(argv[1]) : 0.30;   // seconds per config

    int widths[] = {128, 256, 512, 1024, 2048};
    int depths[] = {0, 1, 2, 4, 8, 16};

    printf("# Forward-pass latency vs WIDTH / DEPTH.  float32, hand-AVX2-FMA, sigmoid every layer.\n");
    printf("# AMD EPYC 7R13, single core @ ~3.6 GHz boost.  net: 768 -> W -> (W->W) x D -> 1\n");
    printf("# weights random (values irrelevant); %.2fs timed per config; denormals flushed.\n", budget);
    printf("# %5s %5s %5s %13s %9s %12s %9s\n",
           "width", "depth", "lyrs", "params", "us/eval", "evals/sec", "GFLOP/s");
    fflush(stdout);

    for (int W : widths) {
        for (int D : depths) {
            Net net; build_net(net, input_dim, W, D, 0x9E3779B97F4A7C15ULL);

            vector<float> in_buf(input_dim);
            std::mt19937 r(7); std::uniform_real_distribution<float> du(-1.f, 1.f);
            for (auto& v : in_buf) v = du(r);

            // Warm caches / branch predictor; feed output back so nothing hoists.
            float last = 0.f;
            for (int i = 0; i < 64; i++) { in_buf[i % input_dim] += last * 1e-3f; last = forward(net, in_buf.data()); }

            long n = 0; double el = 0;
            auto t0 = std::chrono::steady_clock::now();
            while (true) {
                in_buf[n % input_dim] += last * 1e-3f;     // serial dependency: no hoist
                last = forward(net, in_buf.data());
                n++;
                if ((n & 63) == 0) {
                    el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (el >= budget) break;
                }
            }
            volatile float sink = last; (void)sink;        // keep the whole loop live

            double us     = el * 1e6 / n;
            double evps   = n / el;
            double gflops = (2.0 * net.macs()) / (us * 1e-6) / 1e9;
            printf("  %5d %5d %5zu %13lld %9.3f %12.0f %9.1f\n",
                   W, D, net.layers.size(), net.params(), us, evps, gflops);
            fflush(stdout);
        }
    }
    return 0;
}
