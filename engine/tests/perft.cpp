// perft.cpp — Move-generation correctness test.
//
// "perft" (performance test / move-path enumeration) counts the total number of
// leaf positions reachable in exactly N plies from a given position, by playing
// EVERY legal move at every level. These counts are famously known to the exact
// integer for standard test positions, computed and cross-checked by the whole
// chess-programming community. If our generator produces the exact same numbers,
// our move generation, make/unmake, check detection, castling, en passant, and
// promotion handling are all correct. A single off-by-one anywhere shows up as a
// mismatch. This is the gate the board code must pass before we trust the engine.

#include "../src/board.h"
#include <cstdio>
#include <chrono>

using namespace chess;

// Recursively count leaf nodes at the given depth.
static uint64_t perft(Board& b, int depth) {
    if (depth == 0) return 1;
    std::vector<Move> moves;
    b.generate_legal(moves);
    if (depth == 1) return moves.size();   // small optimization: no need to recurse
    uint64_t nodes = 0;
    for (const Move& m : moves) {
        b.make_move(m);
        nodes += perft(b, depth - 1);
        b.unmake_move();
    }
    return nodes;
}

struct Case { const char* fen; int depth; uint64_t expected; const char* name; };

int main() {
    init_attacks();

    // Standard perft positions with community-verified exact node counts.
    Case cases[] = {
        // Start position.
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609, "startpos d5"},
        // "Kiwipete" — a famously tricky middlegame with castling, en passant,
        // pins and promotions; the gold-standard movegen stress test.
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603, "kiwipete d4"},
        // Position 3 — endgame with deep en-passant/check interactions.
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083, "pos3 d6"},
        // Position 4 — promotions and pins.
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333, "pos4 d4"},
        // Position 5.
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487, "pos5 d4"},
    };

    bool all_ok = true;
    for (const Case& c : cases) {
        Board b;
        b.set_fen(c.fen);
        auto t0 = std::chrono::steady_clock::now();
        uint64_t got = perft(b, c.depth);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        bool ok = (got == c.expected);
        all_ok &= ok;
        printf("%-14s depth %d : got %12llu  expected %12llu  %s  (%.2fs, %.1fM nps)\n",
               c.name, c.depth, (unsigned long long)got, (unsigned long long)c.expected,
               ok ? "OK" : "*** MISMATCH ***", secs, secs > 0 ? got / secs / 1e6 : 0);
    }
    printf("\nPERFT %s\n", all_ok ? "PASSED — move generation is correct." : "FAILED — movegen bug!");
    return all_ok ? 0 : 1;
}
