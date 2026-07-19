// uci.cpp — the engine's main program, speaking the UCI protocol.
//
// UCI (Universal Chess Interface) is the standard text protocol chess GUIs and
// match runners use to talk to engines. We implement the commands needed to play
// real games and run matches against a reference opponent:
//   uci            -> identify ourselves, list options, reply "uciok"
//   isready        -> reply "readyok" (after loading the brain)
//   ucinewgame     -> reset for a new game
//   position ...   -> set up a board (startpos or a FEN, then a list of moves)
//   go ...         -> search and print "bestmove <move>"  (movetime / depth / nodes)
//   setoption ...  -> set EvalFile (which brain to load) etc.
//   quit           -> exit
//
// A move is printed in UCI's "long algebraic" form: from-square, to-square, and a
// promotion letter if any (e.g. "e2e4", "e7e8q").

#include "board.h"
#include "eval.h"
#include "judge.h"
#include "search.h"
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

using namespace chess;

static std::string sq_name(int sq) {
    std::string s;
    s += char('a' + (sq & 7));
    s += char('1' + (sq >> 3));
    return s;
}

static std::string move_to_uci(const Move& m) {
    std::string s = sq_name(m.from) + sq_name(m.to);
    if (m.promo != NO_PIECE) {
        const char* p = "pnbrqk";
        s += p[m.promo];
    }
    return s;
}

// Find the legal move on `b` whose UCI string equals `uci` (so "position ... moves"
// can be replayed). Returns false if none matches.
static bool parse_move(Board& b, const std::string& uci, Move& out) {
    std::vector<Move> moves;
    b.generate_legal(moves);
    for (const Move& m : moves)
        if (move_to_uci(m) == uci) { out = m; return true; }
    return false;
}

int main(int argc, char** argv) {
    init_attacks();

    Board board;
    Judge net;                   // the judge: auto-detects shallow vs deep by file magic
    TT tt;                       // shared transposition table (persists across moves)
    int threads = 1;             // Lazy-SMP thread count (set via setoption Threads)
    int hash_mb = 64;
    std::string eval_file = "models/compact_champion.qbin";
    bool net_loaded = false;
    tt.resize_mb(hash_mb);

    auto ensure_net = [&]() {
        if (!net_loaded) {
            net_loaded = net.load(eval_file);
            if (!net_loaded) fprintf(stderr, "could not load eval file %s\n", eval_file.c_str());
        }
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "uci") {
            std::cout << "id name OurEngine\n";
            std::cout << "id author swarajva\n";
            // Advertise the options match runners may set.
            std::cout << "option name EvalFile type string default " << eval_file << "\n";
            std::cout << "option name Hash type spin default 16 min 1 max 1024\n";
            std::cout << "option name Threads type spin default 1 min 1 max 64\n";
            std::cout << "uciok\n" << std::flush;
        } else if (cmd == "isready") {
            // Don't force the net to load here. Match runners (e.g. fastchess) send
            // the first `isready` BEFORE `setoption name EvalFile`, so loading now
            // would try the built-in default path — which is relative and usually
            // wrong under the runner's working directory — and fail noisily. The net
            // loads lazily on the first `go`, by which point EvalFile has been set.
            std::cout << "readyok\n" << std::flush;
        } else if (cmd == "setoption") {
            // Format: setoption name <Name> value <Value>
            std::string tok, name, value;
            ss >> tok;                       // "name"
            ss >> name;
            ss >> tok;                       // "value"
            std::getline(ss, value);
            if (!value.empty() && value[0] == ' ') value.erase(0, 1);
            if (name == "EvalFile") { eval_file = value; net_loaded = false; }
            else if (name == "Threads") { threads = std::max(1, atoi(value.c_str())); }
            else if (name == "Hash") { hash_mb = std::max(1, atoi(value.c_str())); tt.resize_mb(hash_mb); }
        } else if (cmd == "ucinewgame") {
            board.set_startpos();
            tt.clear();          // fresh table for a new game
        } else if (cmd == "position") {
            std::string tok;
            ss >> tok;
            if (tok == "startpos") {
                board.set_startpos();
                ss >> tok;                   // maybe "moves"
            } else if (tok == "fen") {
                // Read the 6 FEN fields back into one string.
                std::string fen, part;
                for (int i = 0; i < 6 && (ss >> part); ++i) fen += (i ? " " : "") + part;
                board.set_fen(fen);
                ss >> tok;                   // maybe "moves"
            }
            if (tok == "moves") {
                std::string mv;
                while (ss >> mv) {
                    Move m;
                    if (parse_move(board, mv, m)) board.make_move(m);
                }
            }
        } else if (cmd == "go") {
            ensure_net();
            SearchLimits lim;
            std::string tok;
            // Parse the time controls we support.
            int wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0, depth = 0;
            long nodes = 0;
            while (ss >> tok) {
                if (tok == "movetime") ss >> movetime;
                else if (tok == "depth") ss >> depth;
                else if (tok == "nodes") ss >> nodes;
                else if (tok == "wtime") ss >> wtime;
                else if (tok == "btime") ss >> btime;
                else if (tok == "winc") ss >> winc;
                else if (tok == "binc") ss >> binc;
            }
            if (movetime > 0) lim.movetime = movetime / 1000.0;
            else if (depth > 0) { lim.max_depth = depth; lim.movetime = 1e9; }
            else if (nodes > 0) { lim.max_nodes = nodes; lim.movetime = 1e9; }
            else if (wtime > 0 || btime > 0) {
                // Simple clock budget: spend ~1/25th of our remaining time plus a
                // bit of the increment. Crude but safe; refined later.
                int my_time = (board.stm == WHITE) ? wtime : btime;
                int my_inc = (board.stm == WHITE) ? winc : binc;
                lim.movetime = (my_time / 25.0 + my_inc * 0.5) / 1000.0;
                if (lim.movetime < 0.01) lim.movetime = 0.01;
            }
            lim.threads = threads;
            SearchResult r = think_parallel(&net, &tt, board, lim);
            // Report a UCI info line (depth, score, nodes) then the chosen move.
            std::cout << "info depth " << r.depth << " score cp " << r.score
                      << " nodes " << r.nodes << "\n";
            std::cout << "bestmove " << move_to_uci(r.best) << "\n" << std::flush;
        } else if (cmd == "quit") {
            break;
        }
    }
    (void)argc; (void)argv;
    return 0;
}
