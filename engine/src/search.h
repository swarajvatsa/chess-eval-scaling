// search.h — Alpha-beta search with modern pruning, plus parallel (Lazy SMP) play.
//
// Single-thread search looks ahead and the brain scores leaves. To use all 64
// cores we run many search threads at once (Lazy SMP): each thread searches the
// SAME root position but, because they share one transposition table and start
// with slightly staggered depths, they naturally explore different lines and feed
// each other's discoveries through that shared table. It is not a clean 64x — the
// pruning is inherently sequential — but it reliably buys depth and Elo.

#pragma once
#include "board.h"
#include "eval.h"
#include "judge.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace chess {

constexpr int INF = 1000000;
constexpr int MATE = 900000;
constexpr int MATE_IN_MAX = MATE - 1000;
constexpr int MAX_PLY = 128;

struct SearchLimits {
    int max_depth = 64;
    double movetime = 1.0;
    long max_nodes = 0;
    int threads = 1;            // number of Lazy-SMP search threads
};

struct SearchResult {
    Move best;
    int score;
    int depth;
    long nodes;
};

enum TTFlag : uint8_t { TT_NONE = 0, TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3 };

// A transposition-table slot, made safe for concurrent threads with the "XOR
// trick": we store `key ^ packed_data` in `key_xor`. A reader recomputes the XOR
// and only trusts the entry if it matches — so a half-written (torn) entry from
// another thread is detected and ignored rather than used as a bogus result. No
// locks needed, which is what keeps Lazy SMP fast.
struct TTEntry {
    std::atomic<uint64_t> key_xor;   // key XOR data
    std::atomic<uint64_t> data;      // packed: score | depth | flag | move
};

struct TTProbe { int score; int depth; uint8_t flag; Move best; };

class TT {
public:
    std::vector<TTEntry> table;
    size_t mask = 0;
    void resize_mb(int mb);
    void clear();
    bool probe(uint64_t key, TTProbe& out) const;
    void store(uint64_t key, int score, int depth, uint8_t flag, const Move& best);
};

// Per-thread search state. The TT and the stop flag are SHARED (pointers/refs);
// killers, history, board copy, and node count are private to each thread so
// threads don't stomp on each other.
class Worker {
public:
    const Judge* net;            // the judge (shallow or deep); named `net` for brevity
    TT* tt;
    std::atomic<bool>* stop;       // shared: set when time is up or another thread says stop
    std::chrono::steady_clock::time_point deadline;
    long nodes;
    Move killers[MAX_PLY][2];
    int history[64][64];
    Move best_root_move;
    int best_root_score;
    int completed_depth;

    Worker(const Judge* n, TT* t, std::atomic<bool>* s) : net(n), tt(t), stop(s), nodes(0) {
        clear_heuristics();
    }
    void clear_heuristics();
    // Run iterative deepening on a private copy of the board until stop/limit.
    void search(Board b, const SearchLimits& lim, int thread_id);

private:
    int negamax(Board& b, int depth, int alpha, int beta, int ply);
    int quiescence(Board& b, int alpha, int beta, int ply);
    bool time_up();
    void score_moves(const Board& b, std::vector<Move>& moves, const Move& tt_move, int ply,
                     std::vector<int>& scores);
};

// Top-level driver: spawn `lim.threads` workers on the position, wait for the
// time limit, and return the best result found by any thread.
SearchResult think_parallel(const Judge* net, TT* tt, Board& b, const SearchLimits& lim);

}  // namespace chess
