// parity_deep.cpp — Verify the C++ integer DEEP judge matches the Python golden.
//
// Reads parity_deep.txt (FEN<TAB>expected_int_score, from make_parity_deep.py),
// loads the same .qbin (CHESSDP1), evaluates each FEN with evaluate_deep, and
// checks every score matches to the integer. Zero mismatches = the C++ deep eval
// is a faithful copy of the trained+quantized deep judge and safe to search with.

#include "../src/board.h"
#include "../src/eval_deep.h"
#include <cstdio>
#include <fstream>
#include <string>

using namespace chess;

int main(int argc, char** argv) {
    init_attacks();
    std::string qbin = (argc > 1) ? argv[1] : "models/deep_b4.qbin";
    std::string parity = (argc > 2) ? argv[2] : "parity_deep.txt";

    DeepNet net;
    if (!net.load(qbin)) { printf("FAILED to load %s\n", qbin.c_str()); return 1; }
    printf("loaded deep net  in=%d dim=%d blocks=%d hidden=%d  scale=%d act_max=%d\n",
           net.in_size, net.dim, net.blocks, net.hidden, net.weight_scale, net.act_max);

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
        int got = evaluate_deep(net, b);
        total++;
        if (got != expected) {
            mismatch++;
            int d = got - expected; if (d < 0) d = -d;
            if (d > max_abs_diff) max_abs_diff = d;
            if (mismatch <= 5)
                printf("  MISMATCH: got %d expected %d  fen=%s\n", got, expected, fen.c_str());
        }
    }
    printf("parity_deep: %d positions, %d mismatches (max abs diff %d)\n",
           total, mismatch, max_abs_diff);
    printf("PARITY %s\n", mismatch == 0 ? "PASSED — C++ deep brain == Python golden." : "FAILED");
    return mismatch == 0 ? 0 : 1;
}
