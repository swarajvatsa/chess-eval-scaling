// parity.cpp — Verify the C++ integer brain matches the Python golden exactly.
//
// Reads parity.txt (FEN<TAB>expected_score lines produced by make_parity.py),
// loads the same .qbin weights, evaluates each FEN with the C++ integer brain,
// and checks every score matches to the integer. If all match, the C++ eval is a
// faithful copy of the trained+quantized net and we can trust it inside search.

#include "../src/board.h"
#include "../src/eval.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using namespace chess;

int main(int argc, char** argv) {
    init_attacks();
    std::string qbin = (argc > 1) ? argv[1] : "models/compact_champion.qbin";
    std::string parity = (argc > 2) ? argv[2] : "parity.txt";

    Net net;
    if (!net.load(qbin)) { printf("FAILED to load %s\n", qbin.c_str()); return 1; }
    printf("loaded net %d->%d->%d  scale=%d act_max=%d\n",
           net.in_size, net.hidden, net.hidden2, net.weight_scale, net.act_max);

    std::ifstream in(parity);
    if (!in) { printf("cannot open %s\n", parity.c_str()); return 1; }

    std::string line;
    int total = 0, mismatch = 0, max_abs_diff = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t tab = line.rfind('\t');
        std::string fen = line.substr(0, tab);
        int expected = std::stoi(line.substr(tab + 1));
        Board b;
        b.set_fen(fen);
        int got = evaluate(net, b);
        total++;
        if (got != expected) {
            mismatch++;
            int d = got - expected; if (d < 0) d = -d;
            if (d > max_abs_diff) max_abs_diff = d;
            if (mismatch <= 5)
                printf("  MISMATCH: got %d expected %d  fen=%s\n", got, expected, fen.c_str());
        }
    }
    printf("parity: %d positions, %d mismatches (max abs diff %d)\n", total, mismatch, max_abs_diff);
    printf("PARITY %s\n", mismatch == 0 ? "PASSED — C++ brain == Python golden." : "FAILED");
    return mismatch == 0 ? 0 : 1;
}
