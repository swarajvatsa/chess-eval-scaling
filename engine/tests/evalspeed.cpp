// evalspeed.cpp — Measure pure judge throughput (evals/sec) as a function of how
// many pieces are on the board. NO search — just: load a position, run the eval a
// few hundred thousand times, time it. This isolates the JUDGE's cost from the
// explorer's, so we can see how much of "endgames think deeper" is the eval getting
// cheaper (our first layer sums one entry per piece, so fewer pieces = fewer adds).
//
// Input: a text file of FEN-per-line. Output: "pieces<TAB>us_per_eval" per line.

#include "../src/board.h"
#include "../src/judge.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <chrono>

using namespace chess;

int main(int argc, char** argv) {
    init_attacks();
    std::string qbin   = (argc > 1) ? argv[1] : "models/compact_champion.qbin";
    std::string fenfile = (argc > 2) ? argv[2] : "/tmp/c_fens.txt";
    int iters = (argc > 3) ? atoi(argv[3]) : 100000;

    Judge net;
    if (!net.load(qbin)) { printf("FAILED to load %s\n", qbin.c_str()); return 1; }

    std::ifstream in(fenfile);
    if (!in) { printf("cannot open %s\n", fenfile.c_str()); return 1; }

    std::string fen;
    while (std::getline(in, fen)) {
        if (fen.empty()) continue;
        Board b;
        b.set_fen(fen);
        int pieces = 0;
        for (int s = 0; s < 64; ++s) if (b.board[s] != NO_PIECE) pieces++;

        // Warm the caches/branch predictor, then time `iters` evals.
        volatile long sink = 0;
        for (int i = 0; i < 1000; ++i) sink += net.eval(b);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) sink += net.eval(b);
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        printf("%d\t%.4f\n", pieces, us);
        fflush(stdout);
    }
    return 0;
}
