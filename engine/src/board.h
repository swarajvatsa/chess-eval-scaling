// board.h — The chess board, pieces, moves, and move generation.
//
// ---------------------------------------------------------------------------
// HOW WE REPRESENT THE BOARD: bitboards
// ---------------------------------------------------------------------------
// A chess board has 64 squares. A "bitboard" is a single 64-bit integer where
// each bit answers a yes/no question about one square. For example, the bitboard
// of "white pawns" has a 1 on every square that holds a white pawn and 0
// everywhere else. We keep one bitboard per (color, piece-type), so 12 in total,
// plus handy combined ones (all white pieces, all black, all occupied).
//
// Why bitboards: questions like "which squares do all my rooks attack?" become a
// few integer operations on these 64-bit words instead of loops over squares.
// Computers do 64-bit integer ops blindingly fast, so this is how every strong
// engine represents the board.
//
// Square numbering: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63. So
//   file (column a..h) = square & 7      (the low 3 bits)
//   rank (row  1..8)   = square >> 3      (the high bits)
// Bit i of a bitboard corresponds to square i.
// ---------------------------------------------------------------------------

#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace chess {

using Bitboard = uint64_t;
using Square = int;   // 0..63
using Color = int;    // 0 = white, 1 = black

enum { WHITE = 0, BLACK = 1 };
// Piece types. Order chosen to match our trainer's channel order (pawn..king).
enum { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5, NO_PIECE = 6 };

// A move packed into a 32-bit int: from (6 bits) | to (6) | promo piece (3) | flags.
// Keeping it small makes move lists cache-friendly.
struct Move {
    int from, to;        // squares 0..63
    int promo;           // promotion piece type (KNIGHT..QUEEN), or NO_PIECE
    int flag;            // see MoveFlag below
    bool operator==(const Move& o) const {
        return from == o.from && to == o.to && promo == o.promo && flag == o.flag;
    }
};

enum MoveFlag {
    NORMAL = 0,
    CAPTURE = 1,
    DOUBLE_PAWN = 2,     // pawn moved two squares (sets en-passant target)
    EN_PASSANT = 3,      // capturing en passant
    CASTLE_K = 4,        // king-side castle
    CASTLE_Q = 5,        // queen-side castle
};

// Castling-rights bit flags, OR'd together in Board::castling.
enum { WK = 1, WQ = 2, BK = 4, BQ = 8 };

// Everything needed to UNDO a move (restore the previous state exactly). We push
// one of these on a stack in make_move and pop it in unmake_move, so search can
// explore a line and rewind cheaply without copying the whole board.
struct Undo {
    Move move;
    int captured;        // piece type captured (NO_PIECE if none)
    int captured_sq;     // square the captured piece sat on (differs for en passant)
    int castling;        // castling rights before the move
    int ep_square;       // en-passant target square before the move (-1 if none)
    int halfmove;        // fifty-move counter before the move
    uint64_t hash;       // Zobrist hash before the move
};

class Board {
public:
    Bitboard pieces[2][6];   // pieces[color][type]
    Bitboard occ[2];         // all pieces of each color
    Bitboard occ_all;        // all pieces
    int board[64];           // piece type on each square (NO_PIECE if empty)
    int color_on[64];        // color on each square (-1 if empty)
    Color stm;               // side to move
    int castling;            // OR of WK/WQ/BK/BQ
    int ep_square;           // en-passant target square, or -1
    int halfmove;            // halfmove clock (for fifty-move rule)
    uint64_t hash;           // Zobrist hash of the position

    std::vector<Undo> history;  // stack for unmake_move

    Board() { set_startpos(); }

    void set_startpos();
    void set_fen(const std::string& fen);

    // Generate all PSEUDO-LEGAL moves (legal except they may leave the king in
    // check). The search filters illegal ones via make/unmake + king-safety, or
    // we filter here — see generate_legal.
    void generate_pseudo(std::vector<Move>& out) const;
    // Generate only fully-LEGAL moves (king not left in check). Used at the root
    // and for perft correctness.
    void generate_legal(std::vector<Move>& out);

    void make_move(const Move& m);
    void unmake_move();

    // "Null move": pass the turn to the opponent without moving a piece. Used by
    // null-move pruning in the search. Keeps the Zobrist hash correct (XORs the
    // side-to-move and clears any en-passant term) so TT lookups stay valid.
    void make_null();
    void unmake_null();

    // Is `sq` attacked by `by_color`? (Used for check detection and castling.)
    bool is_attacked(Square sq, Color by_color) const;
    // Is the side-to-move's king currently in check?
    bool in_check() const;
    // Square of `color`'s king.
    Square king_sq(Color color) const;

    std::string fen() const;

private:
    void put_piece(int color, int type, int sq);
    void remove_piece(int color, int type, int sq);
    void recompute_aggregates();
};

// Initialize the static attack tables (knight/king jumps, sliding rays). Must be
// called once at program start before any move generation.
void init_attacks();

}  // namespace chess
