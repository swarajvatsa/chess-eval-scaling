// acc_difftest.cpp — Prove the INCREMENTAL accumulator equals the FROM-SCRATCH eval,
// bit-for-bit, over thousands of random positions. This is the C++ twin of the
// Python differential test (king_incremental_proto.py) and the gate that lets us
// trust the 100x speedup inside the search.
//
// For each random game we maintain two accumulators (white's and black's
// perspective). After each move we update them with the SAME delta logic the real
// engine will use — patch the few changed feature columns, or refresh a perspective
// from scratch when its own king moved — then check the eval computed from the
// accumulators equals evaluate_king_scratch on the new position. Any divergence is
// a bug we must fix before shipping the accumulator.

#include "../src/board.h"
#include "../src/eval_king.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace chess;

// One perspective's feature index for a piece, mirroring king_perspective_indices /
// king_encode.py. king_sq/piece_sq are oriented (^56 for Black). Kings are not
// feature pieces. Returns -1 if the piece is a king (caller skips).
static int feat(Color persp, int king_sq, int piece_sq, int ptype, bool is_own) {
    if (ptype == KING) return -1;
    bool mirror = (persp == BLACK);
    int k = mirror ? (king_sq ^ 56) : king_sq;
    int s = mirror ? (piece_sq ^ 56) : piece_sq;
    int kind = ptype + (is_own ? 0 : 5);   // PAWN..QUEEN = 0..4
    return (k * 64 + s) * 10 + kind;
}

// Apply a move's deltas to one perspective's accumulator, using ONLY pre-move board
// state + the move (exactly what the engine has at make_move time). Returns true if
// it patched incrementally, false if the perspective must be refreshed (its own
// king moved). Mirrors king_incremental_proto.deltas_for_move.
static bool apply_move_to_persp(const KingNet& net, const Board& pre, Color persp,
                                const Move& m, Accumulator& acc) {
    int king_sq = pre.king_sq(persp);
    int moving = pre.board[m.from];
    int moving_color = pre.color_on[m.from];

    // Own king move (incl. castling) changes K -> every feature changes -> refresh.
    if (moving == KING && moving_color == persp)
        return false;

    bool is_enemy_king_move = (moving == KING && moving_color != persp);

    // Captured piece (normal or en passant).
    int cap_sq = -1, cap_type = NO_PIECE, cap_color = -1;
    if (m.flag == EN_PASSANT) {
        cap_sq = (pre.stm == WHITE) ? m.to - 8 : m.to + 8;
        cap_type = pre.board[cap_sq]; cap_color = pre.color_on[cap_sq];
    } else if (pre.board[m.to] != NO_PIECE) {
        cap_sq = m.to; cap_type = pre.board[m.to]; cap_color = pre.color_on[m.to];
    }

    // The moving piece leaves from (unless it's the enemy king, not a feature).
    if (!is_enemy_king_move) {
        int f = feat(persp, king_sq, m.from, moving, moving_color == persp);
        if (f >= 0) acc_apply_feature(net, acc, f, -1);
    }
    // Captured piece leaves (kings are never captured in legal play, but guard).
    if (cap_type != NO_PIECE && cap_type != KING) {
        int f = feat(persp, king_sq, cap_sq, cap_type, cap_color == persp);
        if (f >= 0) acc_apply_feature(net, acc, f, -1);
    }
    // The moving piece arrives at to (with promotion), unless enemy king.
    if (!is_enemy_king_move) {
        int arrived = (m.promo != NO_PIECE) ? m.promo : moving;
        int f = feat(persp, king_sq, m.to, arrived, moving_color == persp);
        if (f >= 0) acc_apply_feature(net, acc, f, +1);
    }
    // Castling moves the rook too (only reached for the OTHER perspective, since the
    // castling side's own king move returns false above).
    if (m.flag == CASTLE_K || m.flag == CASTLE_Q) {
        int rank_base = (moving_color == WHITE) ? 0 : 56;
        int rf, rt;
        if (m.flag == CASTLE_K) { rf = rank_base + 7; rt = rank_base + 5; }
        else                    { rf = rank_base + 0; rt = rank_base + 3; }
        int f1 = feat(persp, king_sq, rf, ROOK, moving_color == persp);
        int f2 = feat(persp, king_sq, rt, ROOK, moving_color == persp);
        if (f1 >= 0) acc_apply_feature(net, acc, f1, -1);
        if (f2 >= 0) acc_apply_feature(net, acc, f2, +1);
    }
    return true;
}

// A tiny deterministic RNG (so the test is reproducible without <random> overhead).
static uint64_t rng_state = 88172645463325252ULL;
static uint64_t xrng() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
                         rng_state ^= rng_state << 17; return rng_state; }

int main(int argc, char** argv) {
    init_attacks();
    std::string qbin = (argc > 1) ? argv[1] : "models/king.qbin";
    int games = (argc > 2) ? atoi(argv[2]) : 200;
    int max_plies = (argc > 3) ? atoi(argv[3]) : 120;

    KingNet net;
    if (!net.load(qbin)) { printf("FAILED to load %s\n", qbin.c_str()); return 1; }
    printf("loaded king net first_dim=%d blocks=%d hidden=%d\n",
           net.first_dim, net.blocks, net.hidden);

    long checks = 0, refreshes = 0, patches = 0;
    for (int g = 0; g < games; ++g) {
        Board b;
        b.set_startpos();
        Accumulator wacc, bacc;
        acc_refresh(net, b, WHITE, wacc);
        acc_refresh(net, b, BLACK, bacc);
        for (int ply = 0; ply < max_plies; ++ply) {
            std::vector<Move> moves;
            b.generate_legal(moves);
            if (moves.empty()) break;
            Move m = moves[xrng() % moves.size()];

            // Update each perspective's accumulator from PRE-move state.
            bool wpatch = apply_move_to_persp(net, b, WHITE, m, wacc);
            bool bpatch = apply_move_to_persp(net, b, BLACK, m, bacc);

            b.make_move(m);

            // Perspectives whose own king moved must be refreshed post-move.
            if (!wpatch) { acc_refresh(net, b, WHITE, wacc); refreshes++; } else patches++;
            if (!bpatch) { acc_refresh(net, b, BLACK, bacc); refreshes++; } else patches++;

            int inc = evaluate_king_from_acc(net, b.stm, wacc, bacc);
            int scratch = evaluate_king_scratch(net, b);
            if (inc != scratch) {
                printf("  MISMATCH game %d ply %d: incremental %d != scratch %d (move %d->%d)\n",
                       g, ply, inc, scratch, m.from, m.to);
                printf("ACC DIFFTEST FAILED\n");
                return 2;
            }
            checks++;
        }
    }
    printf("acc difftest: %ld checks PASSED, %ld patches, %ld refreshes over %d games\n",
           checks, patches, refreshes, games);
    printf("ACC DIFFTEST PASSED — incremental accumulator == from-scratch, bit-exact.\n");
    return 0;
}
