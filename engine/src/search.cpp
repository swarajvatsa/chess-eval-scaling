// search.cpp — CLEAN alpha-beta minimax + Lazy SMP.
//
// Design rule for this file: we only keep tricks that NEVER change the answer the
// search would have given by looking at every move to the full depth. Those are:
//   - the transposition table (remember a position we already solved),
//   - move ordering (try the best-looking move first so cutoffs happen sooner),
//   - principal-variation search (probe later moves with a zero-width window first;
//     if one surprises us we re-search it at full width — same result, less work),
//   - quiescence at the leaves (keep looking while captures are flying so we don't
//     stop in the middle of a trade),
//   - the check extension (look one ply deeper when in check — that's looking MORE,
//     never less).
// We deliberately do NOT use the two "gambles" that skip work which could matter:
//   - null-move pruning (let the opponent move twice and, if we're still winning,
//     assume the whole branch is safe) — REMOVED.
//   - late-move reductions (search moves that look boring to a shallower depth) —
//     REMOVED.
// So every legal move is searched to the full depth. Slower per node, but it is the
// true minimax value with no bets.

#include "search.h"
#include <algorithm>
#include <cstring>

namespace chess {

// ---------------------------------------------------------------------------
// Transposition table (lock-free via the XOR trick).
//
// We pack the result (score, depth, flag, move) into one 64-bit word `data`, and
// store key_xor = hash ^ data. To read: load both words, XOR them, and check the
// result equals the hash we're looking for. If another thread wrote only one of
// the two words (a "torn" write), the XOR won't match and we safely ignore the
// entry. This gives correct concurrent behavior with no locks.
// ---------------------------------------------------------------------------
static inline uint64_t pack(int score, int depth, uint8_t flag, const Move& m) {
    // score: 32 bits (offset to stay non-negative) | depth: 8 | flag: 2 | move: 22
    uint64_t s = (uint32_t)(score + (1 << 30));
    uint64_t d = (uint8_t)(depth & 0xFF);
    uint64_t f = (flag & 0x3);
    uint64_t mv = ((uint64_t)(m.from & 0x3F)) | ((uint64_t)(m.to & 0x3F) << 6)
                | ((uint64_t)(m.promo & 0x7) << 12) | ((uint64_t)(m.flag & 0x7) << 15);
    return (s) | (d << 32) | (f << 40) | (mv << 42);
}
static inline void unpack(uint64_t w, TTProbe& o) {
    o.score = (int)((int64_t)(w & 0xFFFFFFFF) - (1 << 30));
    o.depth = (int)((w >> 32) & 0xFF);
    o.flag = (uint8_t)((w >> 40) & 0x3);
    uint64_t mv = w >> 42;
    o.best.from = (int)(mv & 0x3F);
    o.best.to = (int)((mv >> 6) & 0x3F);
    o.best.promo = (int)((mv >> 12) & 0x7);
    o.best.flag = (int)((mv >> 15) & 0x7);
}

void TT::resize_mb(int mb) {
    size_t bytes = (size_t)mb * 1024 * 1024;
    size_t n = 1;
    while (n * sizeof(TTEntry) * 2 <= bytes) n <<= 1;
    table = std::vector<TTEntry>(n);     // atomics default-init to 0
    mask = n - 1;
}
void TT::clear() {
    for (auto& e : table) { e.key_xor.store(0, std::memory_order_relaxed);
                            e.data.store(0, std::memory_order_relaxed); }
}
bool TT::probe(uint64_t key, TTProbe& out) const {
    const TTEntry& e = table[key & mask];
    uint64_t kx = e.key_xor.load(std::memory_order_relaxed);
    uint64_t d  = e.data.load(std::memory_order_relaxed);
    if ((kx ^ d) != key) return false;   // empty or torn write -> miss
    unpack(d, out);
    return true;
}
void TT::store(uint64_t key, int score, int depth, uint8_t flag, const Move& best) {
    TTEntry& e = table[key & mask];
    // Depth-preferred: don't overwrite a clearly deeper exact entry for the same key.
    uint64_t kx = e.key_xor.load(std::memory_order_relaxed);
    uint64_t od = e.data.load(std::memory_order_relaxed);
    if ((kx ^ od) == key) {
        TTProbe cur; unpack(od, cur);
        if (cur.depth > depth && cur.flag == TT_EXACT) return;
    }
    uint64_t data = pack(score, depth, flag, best);
    e.data.store(data, std::memory_order_relaxed);
    e.key_xor.store(key ^ data, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
void Worker::clear_heuristics() {
    std::memset(killers, 0, sizeof(killers));
    std::memset(history, 0, sizeof(history));
}

bool Worker::time_up() {
    if ((nodes & 2047) == 0)
        if (std::chrono::steady_clock::now() >= deadline) stop->store(true);
    return stop->load(std::memory_order_relaxed);
}

static const int PIECE_VAL[6] = {100, 320, 330, 500, 900, 0};

void Worker::score_moves(const Board& b, std::vector<Move>& moves, const Move& tt_move,
                         int ply, std::vector<int>& scores) {
    scores.resize(moves.size());
    for (size_t i = 0; i < moves.size(); ++i) {
        const Move& m = moves[i];
        int s;
        if (m == tt_move) s = 2000000;
        else if (m.flag == CAPTURE || m.flag == EN_PASSANT) {
            int victim = (m.flag == EN_PASSANT) ? PAWN : b.board[m.to];
            int attacker = b.board[m.from];
            s = 1000000 + 10 * PIECE_VAL[victim] - PIECE_VAL[attacker];
        } else if (m.promo != NO_PIECE) s = 900000 + PIECE_VAL[m.promo];
        else if (m == killers[ply][0]) s = 800000;
        else if (m == killers[ply][1]) s = 700000;
        else s = history[m.from][m.to];
        scores[i] = s;
    }
}

static void pick_move(std::vector<Move>& moves, std::vector<int>& scores, size_t i) {
    size_t best = i;
    for (size_t j = i + 1; j < moves.size(); ++j)
        if (scores[j] > scores[best]) best = j;
    std::swap(moves[i], moves[best]);
    std::swap(scores[i], scores[best]);
}

int Worker::quiescence(Board& b, int alpha, int beta, int ply) {
    nodes++;
    int stand = net->eval(b);
    if (stand >= beta) return beta;
    if (stand > alpha) alpha = stand;
    if (time_up() || ply >= MAX_PLY - 1) return alpha;

    std::vector<Move> moves;
    b.generate_pseudo(moves);
    Move none{0, 0, NO_PIECE, NORMAL};
    std::vector<int> scores;
    score_moves(b, moves, none, ply, scores);

    Color us = b.stm;
    for (size_t i = 0; i < moves.size(); ++i) {
        pick_move(moves, scores, i);
        const Move& m = moves[i];
        if (m.flag != CAPTURE && m.flag != EN_PASSANT && m.promo == NO_PIECE) continue;
        b.make_move(m);
        if (b.is_attacked(b.king_sq(us), b.stm)) { b.unmake_move(); continue; }
        int score = -quiescence(b, -beta, -alpha, ply + 1);
        b.unmake_move();
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
        if (stop->load(std::memory_order_relaxed)) return alpha;
    }
    return alpha;
}

int Worker::negamax(Board& b, int depth, int alpha, int beta, int ply) {
    nodes++;
    if (time_up()) return 0;
    if (b.halfmove >= 100) return 0;
    if (ply >= MAX_PLY - 1) return net->eval(b);

    bool in_chk = b.in_check();
    if (in_chk) depth++;                       // check extension
    if (depth <= 0) return quiescence(b, alpha, beta, ply);

    bool pv_node = (beta - alpha) > 1;

    Move tt_move{0, 0, NO_PIECE, NORMAL};
    TTProbe e;
    if (tt->probe(b.hash, e)) {
        tt_move = e.best;
        if (!pv_node && e.depth >= depth) {
            if (e.flag == TT_EXACT) return e.score;
            if (e.flag == TT_LOWER && e.score >= beta) return e.score;
            if (e.flag == TT_UPPER && e.score <= alpha) return e.score;
        }
    }

    std::vector<Move> moves;
    b.generate_pseudo(moves);
    std::vector<int> scores;
    score_moves(b, moves, tt_move, ply, scores);

    Color us = b.stm;
    int legal = 0, best = -INF, orig_alpha = alpha;
    Move best_move{0, 0, NO_PIECE, NORMAL};

    for (size_t i = 0; i < moves.size(); ++i) {
        pick_move(moves, scores, i);
        const Move& m = moves[i];
        b.make_move(m);
        if (b.is_attacked(b.king_sq(us), b.stm)) { b.unmake_move(); continue; }
        legal++;
        bool is_quiet = (m.flag != CAPTURE && m.flag != EN_PASSANT && m.promo == NO_PIECE);
        int score;

        // Clean minimax with principal-variation search (PVS). PVS is NOT a gamble:
        // every move is still searched to the full depth. The only optimization is
        // the WINDOW we use. We trust our move ordering enough to assume the first
        // move is best, so for every later move we first ask the cheap yes/no
        // question "can this beat what we already have?" with a zero-width window.
        // If the answer is no (the usual case) we saved time. If the answer is a
        // surprise yes, we immediately RE-SEARCH that move at full width to get its
        // true value. Same answer as plain alpha-beta, just fewer wasted nodes.
        if (legal == 1) {
            score = -negamax(b, depth - 1, -beta, -alpha, ply + 1);
        } else {
            score = -negamax(b, depth - 1, -alpha - 1, -alpha, ply + 1);
            if (score > alpha && score < beta)
                score = -negamax(b, depth - 1, -beta, -alpha, ply + 1);
        }
        b.unmake_move();
        if (stop->load(std::memory_order_relaxed)) return best;

        if (score > best) { best = score; best_move = m; }
        if (best > alpha) alpha = best;
        if (alpha >= beta) {
            if (is_quiet) {
                if (!(m == killers[ply][0])) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
                history[m.from][m.to] += depth * depth;
            }
            break;
        }
    }
    if (legal == 0) return in_chk ? -MATE + ply : 0;
    uint8_t flag = (best <= orig_alpha) ? TT_UPPER : (best >= beta ? TT_LOWER : TT_EXACT);
    tt->store(b.hash, best, depth, flag, best_move);
    return best;
}

// One thread's iterative deepening. thread_id staggers the starting depth a touch
// so threads diverge (the essence of Lazy SMP diversification).
void Worker::search(Board b, const SearchLimits& lim, int thread_id) {
    std::vector<Move> root_moves;
    b.generate_legal(root_moves);
    if (root_moves.empty()) { best_root_move = {0,0,NO_PIECE,NORMAL}; completed_depth = 0; return; }
    best_root_move = root_moves[0];
    best_root_score = 0;
    completed_depth = 0;

    Move best_so_far = root_moves[0];
    int start_depth = 1 + (thread_id % 2);    // odd threads start one ply deeper

    for (int depth = start_depth; depth <= lim.max_depth; ++depth) {
        int alpha = -INF, beta = INF, local_best_score = -INF;
        Move local_best = best_so_far;
        std::stable_sort(root_moves.begin(), root_moves.end(),
                         [&](const Move& a, const Move& c) {
                             return (a == best_so_far) > (c == best_so_far);
                         });
        for (size_t i = 0; i < root_moves.size(); ++i) {
            const Move& m = root_moves[i];
            b.make_move(m);
            int score;
            if (i == 0) score = -negamax(b, depth - 1, -beta, -alpha, 1);
            else {
                score = -negamax(b, depth - 1, -alpha - 1, -alpha, 1);
                if (score > alpha) score = -negamax(b, depth - 1, -beta, -alpha, 1);
            }
            b.unmake_move();
            if (stop->load(std::memory_order_relaxed)) break;
            if (score > local_best_score) { local_best_score = score; local_best = m; }
            if (score > alpha) alpha = score;
        }
        if (!stop->load(std::memory_order_relaxed)) {
            best_so_far = local_best;
            best_root_move = local_best;
            best_root_score = local_best_score;
            completed_depth = depth;
        } else break;
        if (std::chrono::steady_clock::now() >= deadline) break;
        if (local_best_score >= MATE_IN_MAX) break;
    }
}

// ---------------------------------------------------------------------------
// Lazy SMP driver: launch N worker threads on the position, wait, pick the best
// result (deepest completed; ties broken by score). The shared TT lets the
// threads help each other; the shared stop flag ends them together at the limit.
// ---------------------------------------------------------------------------
SearchResult think_parallel(const Judge* net, TT* tt, Board& b, const SearchLimits& lim) {
    std::atomic<bool> stop{false};
    auto deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(lim.movetime));

    int n = std::max(1, lim.threads);
    std::vector<Worker> workers;
    workers.reserve(n);
    for (int i = 0; i < n; ++i) {
        workers.emplace_back(net, tt, &stop);
        workers.back().deadline = deadline;
    }

    std::vector<std::thread> pool;
    for (int i = 1; i < n; ++i)
        pool.emplace_back([&, i]() { workers[i].search(b, lim, i); });
    // Run worker 0 on this thread too (no point idling it).
    workers[0].search(b, lim, 0);
    // Time's basically up once worker 0 returns; tell the rest to stop and join.
    stop.store(true);
    for (auto& t : pool) t.join();

    // Choose the best across threads: prefer the deepest completed search.
    SearchResult r;
    r.best = workers[0].best_root_move;
    r.score = workers[0].best_root_score;
    r.depth = workers[0].completed_depth;
    r.nodes = 0;
    for (auto& w : workers) {
        r.nodes += w.nodes;
        if (w.completed_depth > r.depth ||
            (w.completed_depth == r.depth && w.best_root_score > r.score)) {
            r.depth = w.completed_depth;
            r.score = w.best_root_score;
            r.best = w.best_root_move;
        }
    }
    return r;
}

}  // namespace chess
