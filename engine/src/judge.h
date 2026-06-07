// judge.h — One handle that can be ANY of our three judges.
//
// We ship three eval architectures, distinguished by the weight file's 8-byte magic:
//   "CHESSBR1" — the original shallow net (768 -> 1024 -> 256 -> 1)
//   "CHESSDP1" — the deep residual judge (768 -> dim -> N residual blocks -> 1)
//   "CHESSKG1" — the king-relative deep judge (40960x2 -> dim -> tower -> 1)
// Rather than thread three code paths through the whole search, we wrap them in one
// `Judge`. It sniffs the magic at load time and remembers which kind it is; every
// eval call is then a single, perfectly-predicted branch (the kind never changes
// mid-search), so there is no virtual-call cost on the hot path.
//
// The king-relative judge here uses the FROM-SCRATCH eval (both perspectives
// recomputed each call). The incremental accumulator is a separate, search-side
// optimization (it must hook make/unmake, which holds per-thread board state); the
// from-scratch path is correct and is what we ladder first.
//
// The search and UCI layers hold a `const Judge*` and call judge->eval(board).

#pragma once
#include "board.h"
#include "eval.h"
#include "eval_deep.h"
#include "eval_king.h"
#include <cstdio>
#include <cstring>
#include <string>

namespace chess {

enum class JudgeKind { SHALLOW, DEEP, KING };

struct Judge {
    Net shallow;
    DeepNet deep;
    KingNet king;
    JudgeKind kind = JudgeKind::SHALLOW;
    bool loaded = false;

    // DUMMY mode: when set, eval() ignores the net and returns a cheap, position-
    // dependent pseudo-random score derived from the Zobrist hash. This factors the
    // evaluator's COST out entirely so a benchmark can time the pure search machinery
    // (movegen, make/unmake, ordering, alpha-beta, TT). The value still varies per
    // position, so alpha-beta cutoffs and move ordering behave like a real search —
    // we're measuring the explorer, not a degenerate all-equal tree.
    bool dummy = false;

    bool load(const std::string& path) {
        loaded = false;
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "judge: cannot open %s\n", path.c_str()); return false; }
        char magic[8];
        size_t got = fread(magic, 1, 8, f);
        fclose(f);
        if (got != 8) { fprintf(stderr, "judge: %s too short\n", path.c_str()); return false; }

        if (memcmp(magic, "CHESSDP1", 8) == 0) {
            kind = JudgeKind::DEEP;  loaded = deep.load(path);
        } else if (memcmp(magic, "CHESSKG1", 8) == 0) {
            kind = JudgeKind::KING;  loaded = king.load(path);
        } else if (memcmp(magic, "CHESSBR1", 8) == 0) {
            kind = JudgeKind::SHALLOW; loaded = shallow.load(path);
        } else {
            fprintf(stderr, "judge: unknown magic in %s\n", path.c_str());
            return false;
        }
        return loaded;
    }

    // The hot path: score a position from the side-to-move's point of view.
    inline int eval(const Board& b) const {
        if (dummy) {
            // Cheap hash mix (splitmix64 finalizer) -> a score in roughly [-512,512].
            // Deterministic per position, so the search is reproducible.
            uint64_t z = b.hash;
            z ^= z >> 30; z *= 0xbf58476d1ce4e5b9ULL;
            z ^= z >> 27; z *= 0x94d049bb133111ebULL;
            z ^= z >> 31;
            return (int)(z & 1023) - 512;
        }
        switch (kind) {
            case JudgeKind::DEEP: return evaluate_deep(deep, b);
            case JudgeKind::KING: return evaluate_king_scratch(king, b);
            default:              return evaluate(shallow, b);
        }
    }
};

}  // namespace chess
