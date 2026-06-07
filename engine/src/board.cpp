// board.cpp — board setup, attack tables, move generation, make/unmake.
//
// This is the correctness-critical heart of the engine. Every comment here earns
// its place: move generation bugs are the #1 cause of engines that "mostly work"
// but occasionally make or allow illegal moves. We verify the whole file with
// `perft` (counting leaf nodes to a fixed depth and matching known-exact totals).

#include "board.h"
#include <cstring>
#include <sstream>

namespace chess {

// ---------------------------------------------------------------------------
// Bit helpers. A bitboard is 64 bits; bit i == square i.
// ---------------------------------------------------------------------------
static inline Bitboard bb(Square s) { return Bitboard(1) << s; }
static inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
static inline int lsb(Bitboard b) { return __builtin_ctzll(b); }      // index of lowest set bit
static inline int pop_lsb(Bitboard& b) { int s = lsb(b); b &= b - 1; return s; }  // take & clear it
static inline int file_of(Square s) { return s & 7; }
static inline int rank_of(Square s) { return s >> 3; }

// ---------------------------------------------------------------------------
// Static attack tables, filled by init_attacks().
//   KNIGHT_ATK[sq], KING_ATK[sq]  — the squares a knight/king on `sq` attacks.
//   For sliding pieces (bishop/rook/queen) we ray-cast at move-gen time, which is
//   simple and plenty fast for our needs (we can swap in magic bitboards later if
//   movegen ever becomes the bottleneck — it won't; eval dominates).
// ---------------------------------------------------------------------------
static Bitboard KNIGHT_ATK[64];
static Bitboard KING_ATK[64];

// The 8 sliding directions as (file_delta, rank_delta).
static const int BISHOP_DIR[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
static const int ROOK_DIR[4][2]   = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

void init_attacks() {
    // Knight jumps: all 8 (file,rank) offsets of the L-shape.
    const int kn[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
    const int kg[8][2] = {{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1}};
    for (Square s = 0; s < 64; ++s) {
        int f = file_of(s), r = rank_of(s);
        Bitboard n = 0, k = 0;
        for (int i = 0; i < 8; ++i) {
            int nf = f + kn[i][0], nr = r + kn[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) n |= bb(nr * 8 + nf);
            int kf = f + kg[i][0], kr = r + kg[i][1];
            if (kf >= 0 && kf < 8 && kr >= 0 && kr < 8) k |= bb(kr * 8 + kf);
        }
        KNIGHT_ATK[s] = n;
        KING_ATK[s] = k;
    }
}

// Cast a ray from `sq` in the given (df,dr) direction, stopping at (and including)
// the first occupied square. Returns the bitboard of reachable squares. This is
// how we compute bishop/rook/queen attacks given the current occupancy.
static Bitboard ray_attacks(Square sq, int df, int dr, Bitboard occ) {
    Bitboard atk = 0;
    int f = file_of(sq), r = rank_of(sq);
    while (true) {
        f += df; r += dr;
        if (f < 0 || f > 7 || r < 0 || r > 7) break;
        Square t = r * 8 + f;
        atk |= bb(t);
        if (occ & bb(t)) break;   // blocked: include the blocker, then stop
    }
    return atk;
}

static Bitboard bishop_attacks(Square sq, Bitboard occ) {
    Bitboard a = 0;
    for (auto& d : BISHOP_DIR) a |= ray_attacks(sq, d[0], d[1], occ);
    return a;
}
static Bitboard rook_attacks(Square sq, Bitboard occ) {
    Bitboard a = 0;
    for (auto& d : ROOK_DIR) a |= ray_attacks(sq, d[0], d[1], occ);
    return a;
}

// ---------------------------------------------------------------------------
// Board mutation helpers — keep all the redundant representations in sync.
// ---------------------------------------------------------------------------
void Board::put_piece(int color, int type, int sq) {
    pieces[color][type] |= bb(sq);
    board[sq] = type;
    color_on[sq] = color;
}
void Board::remove_piece(int color, int type, int sq) {
    pieces[color][type] &= ~bb(sq);
    board[sq] = NO_PIECE;
    color_on[sq] = -1;
}
void Board::recompute_aggregates() {
    occ[WHITE] = occ[BLACK] = 0;
    for (int t = 0; t < 6; ++t) { occ[WHITE] |= pieces[WHITE][t]; occ[BLACK] |= pieces[BLACK][t]; }
    occ_all = occ[WHITE] | occ[BLACK];
}

// ---------------------------------------------------------------------------
// Zobrist hashing: a position's hash is the XOR of random 64-bit numbers, one per
// (piece,color,square) present, plus terms for side-to-move/castling/en-passant.
// XOR is its own inverse, so making/unmaking a move just XORs the few changed
// terms. The transposition table uses this hash as a position key.
// ---------------------------------------------------------------------------
static uint64_t ZOB_PIECE[2][6][64];
static uint64_t ZOB_STM;
static uint64_t ZOB_CASTLE[16];
static uint64_t ZOB_EP[8];     // keyed by file of the en-passant square
static bool zob_ready = false;

static uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void init_zobrist() {
    if (zob_ready) return;
    uint64_t s = 0xC0FFEE123456789ULL;
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 6; ++t)
            for (int sq = 0; sq < 64; ++sq) ZOB_PIECE[c][t][sq] = splitmix64(s);
    ZOB_STM = splitmix64(s);
    for (int i = 0; i < 16; ++i) ZOB_CASTLE[i] = splitmix64(s);
    for (int i = 0; i < 8; ++i) ZOB_EP[i] = splitmix64(s);
    zob_ready = true;
}

uint64_t compute_hash(const Board& b) {
    uint64_t h = 0;
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 6; ++t) {
            Bitboard x = b.pieces[c][t];
            while (x) { int sq = pop_lsb(x); h ^= ZOB_PIECE[c][t][sq]; }
        }
    if (b.stm == BLACK) h ^= ZOB_STM;
    h ^= ZOB_CASTLE[b.castling];
    if (b.ep_square >= 0) h ^= ZOB_EP[file_of(b.ep_square)];
    return h;
}

// ---------------------------------------------------------------------------
// Position setup
// ---------------------------------------------------------------------------
void Board::set_fen(const std::string& fen) {
    init_zobrist();
    for (int c = 0; c < 2; ++c) for (int t = 0; t < 6; ++t) pieces[c][t] = 0;
    for (int i = 0; i < 64; ++i) { board[i] = NO_PIECE; color_on[i] = -1; }
    castling = 0; ep_square = -1; halfmove = 0;
    history.clear();

    std::istringstream ss(fen);
    std::string placement, side, castle, ep;
    ss >> placement >> side >> castle >> ep;
    // Placement is given rank 8 down to rank 1, files a..h within a rank.
    int rank = 7, file = 0;
    auto type_of = [](char c) {
        switch (c) { case 'p':case 'P':return PAWN; case 'n':case 'N':return KNIGHT;
            case 'b':case 'B':return BISHOP; case 'r':case 'R':return ROOK;
            case 'q':case 'Q':return QUEEN; case 'k':case 'K':return KING; }
        return NO_PIECE;
    };
    for (char c : placement) {
        if (c == '/') { rank--; file = 0; }
        else if (c >= '1' && c <= '8') file += c - '0';
        else { int col = (c >= 'a') ? BLACK : WHITE; put_piece(col, type_of(c), rank * 8 + file); file++; }
    }
    stm = (side == "w") ? WHITE : BLACK;
    if (castle.find('K') != std::string::npos) castling |= WK;
    if (castle.find('Q') != std::string::npos) castling |= WQ;
    if (castle.find('k') != std::string::npos) castling |= BK;
    if (castle.find('q') != std::string::npos) castling |= BQ;
    if (ep != "-" && ep.size() == 2) ep_square = (ep[1]-'1')*8 + (ep[0]-'a');
    recompute_aggregates();
    hash = compute_hash(*this);
}

void Board::set_startpos() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

Square Board::king_sq(Color color) const { return lsb(pieces[color][KING]); }

// ---------------------------------------------------------------------------
// Attack detection: is `sq` attacked by any piece of `by_color`?
// We ask it the other way around — "from sq, do the attack patterns hit an
// enemy piece of the right type?" — which is the standard efficient trick.
// ---------------------------------------------------------------------------
bool Board::is_attacked(Square sq, Color by_color) const {
    // Pawns: a square is attacked by an enemy pawn if an enemy pawn sits where it
    // could capture INTO sq. White pawns capture upward, black downward, so we
    // look in the opposite direction from sq.
    Bitboard pawns = pieces[by_color][PAWN];
    if (by_color == WHITE) {
        // white pawn on sq-7 / sq-9 (one rank below) attacks sq
        if (file_of(sq) > 0 && (pawns & bb(sq - 9))) return true;
        if (file_of(sq) < 7 && (pawns & bb(sq - 7))) return true;
    } else {
        if (file_of(sq) < 7 && (pawns & bb(sq + 9))) return true;
        if (file_of(sq) > 0 && (pawns & bb(sq + 7))) return true;
    }
    if (KNIGHT_ATK[sq] & pieces[by_color][KNIGHT]) return true;
    if (KING_ATK[sq] & pieces[by_color][KING]) return true;
    // Sliding: bishops/queens on diagonals, rooks/queens on lines.
    Bitboard bishops = pieces[by_color][BISHOP] | pieces[by_color][QUEEN];
    if (bishop_attacks(sq, occ_all) & bishops) return true;
    Bitboard rooks = pieces[by_color][ROOK] | pieces[by_color][QUEEN];
    if (rook_attacks(sq, occ_all) & rooks) return true;
    return false;
}

bool Board::in_check() const {
    return is_attacked(king_sq(stm), stm ^ 1);
}

// ---------------------------------------------------------------------------
// Pseudo-legal move generation: produces all moves that follow piece movement
// rules, but may leave our own king in check (those are filtered later).
// ---------------------------------------------------------------------------
void Board::generate_pseudo(std::vector<Move>& out) const {
    Color us = stm, them = stm ^ 1;
    Bitboard own = occ[us], opp = occ[them];

    // ---- Pawns ----
    Bitboard pawns = pieces[us][PAWN];
    int forward = (us == WHITE) ? 8 : -8;
    int start_rank = (us == WHITE) ? 1 : 6;
    int promo_rank = (us == WHITE) ? 7 : 0;
    while (pawns) {
        int from = pop_lsb(pawns);
        int r = rank_of(from);
        int one = from + forward;
        // single push to an empty square
        if (one >= 0 && one < 64 && !(occ_all & bb(one))) {
            if (rank_of(one) == promo_rank) {
                for (int p = QUEEN; p >= KNIGHT; --p) out.push_back({from, one, p, NORMAL});
            } else {
                out.push_back({from, one, NO_PIECE, NORMAL});
                // double push from the starting rank, over an empty intermediate
                if (r == start_rank) {
                    int two = from + 2 * forward;
                    if (!(occ_all & bb(two))) out.push_back({from, two, NO_PIECE, DOUBLE_PAWN});
                }
            }
        }
        // captures (including promotion-captures and en passant)
        for (int dc : {-1, 1}) {
            int nf = file_of(from) + dc;
            if (nf < 0 || nf > 7) continue;
            int to = from + forward + dc;
            if (to < 0 || to > 63) continue;
            if (opp & bb(to)) {
                if (rank_of(to) == promo_rank)
                    for (int p = QUEEN; p >= KNIGHT; --p) out.push_back({from, to, p, CAPTURE});
                else
                    out.push_back({from, to, NO_PIECE, CAPTURE});
            } else if (to == ep_square) {
                // en passant: the captured pawn is beside us, not on `to`
                out.push_back({from, to, NO_PIECE, EN_PASSANT});
            }
        }
    }

    // ---- Knights ----
    Bitboard kn = pieces[us][KNIGHT];
    while (kn) {
        int from = pop_lsb(kn);
        Bitboard t = KNIGHT_ATK[from] & ~own;
        while (t) { int to = pop_lsb(t); out.push_back({from, to, NO_PIECE, (opp & bb(to)) ? CAPTURE : NORMAL}); }
    }
    // ---- Bishops / Rooks / Queens (sliding) ----
    auto gen_slider = [&](Bitboard pcs, bool diag, bool orth) {
        while (pcs) {
            int from = pop_lsb(pcs);
            Bitboard t = 0;
            if (diag) t |= bishop_attacks(from, occ_all);
            if (orth) t |= rook_attacks(from, occ_all);
            t &= ~own;
            while (t) { int to = pop_lsb(t); out.push_back({from, to, NO_PIECE, (opp & bb(to)) ? CAPTURE : NORMAL}); }
        }
    };
    gen_slider(pieces[us][BISHOP], true, false);
    gen_slider(pieces[us][ROOK], false, true);
    gen_slider(pieces[us][QUEEN], true, true);

    // ---- King (normal moves) ----
    int ksq = king_sq(us);
    Bitboard kt = KING_ATK[ksq] & ~own;
    while (kt) { int to = pop_lsb(kt); out.push_back({ksq, to, NO_PIECE, (opp & bb(to)) ? CAPTURE : NORMAL}); }

    // ---- Castling ----
    // Requirements: the right is still available, the squares between king and
    // rook are empty, and the king does not start/pass/end on an attacked square.
    if (us == WHITE) {
        if ((castling & WK) && !(occ_all & (bb(5) | bb(6))) &&
            !is_attacked(4, them) && !is_attacked(5, them) && !is_attacked(6, them))
            out.push_back({4, 6, NO_PIECE, CASTLE_K});
        if ((castling & WQ) && !(occ_all & (bb(1) | bb(2) | bb(3))) &&
            !is_attacked(4, them) && !is_attacked(3, them) && !is_attacked(2, them))
            out.push_back({4, 2, NO_PIECE, CASTLE_Q});
    } else {
        if ((castling & BK) && !(occ_all & (bb(61) | bb(62))) &&
            !is_attacked(60, them) && !is_attacked(61, them) && !is_attacked(62, them))
            out.push_back({60, 62, NO_PIECE, CASTLE_K});
        if ((castling & BQ) && !(occ_all & (bb(57) | bb(58) | bb(59))) &&
            !is_attacked(60, them) && !is_attacked(59, them) && !is_attacked(58, them))
            out.push_back({60, 58, NO_PIECE, CASTLE_Q});
    }
}

void Board::generate_legal(std::vector<Move>& out) {
    std::vector<Move> pseudo;
    pseudo.reserve(64);
    generate_pseudo(pseudo);
    Color us = stm;
    for (const Move& m : pseudo) {
        make_move(m);
        // After making our move it's the opponent's turn; our king must not be
        // attacked. king_sq(us) is our king; opponent is stm now.
        if (!is_attacked(king_sq(us), stm)) out.push_back(m);
        unmake_move();
    }
}

// ---------------------------------------------------------------------------
// make_move / unmake_move: apply a move and keep the hash + undo stack correct.
// ---------------------------------------------------------------------------
void Board::make_move(const Move& m) {
    Undo u;
    u.move = m;
    u.captured = NO_PIECE;
    u.captured_sq = -1;
    u.castling = castling;
    u.ep_square = ep_square;
    u.halfmove = halfmove;
    u.hash = hash;

    Color us = stm, them = stm ^ 1;
    int piece = board[m.from];

    // Remove the old en-passant hash term; we'll add a new one if a double push.
    if (ep_square >= 0) hash ^= ZOB_EP[file_of(ep_square)];
    int new_ep = -1;

    halfmove++;
    if (piece == PAWN) halfmove = 0;  // pawn moves reset the fifty-move clock

    // Handle the captured piece (normal capture or en passant).
    if (m.flag == EN_PASSANT) {
        // The captured pawn sits on the square behind the destination.
        int cap_sq = (us == WHITE) ? m.to - 8 : m.to + 8;
        u.captured = PAWN; u.captured_sq = cap_sq;
        hash ^= ZOB_PIECE[them][PAWN][cap_sq];
        remove_piece(them, PAWN, cap_sq);
        halfmove = 0;
    } else if (board[m.to] != NO_PIECE) {
        u.captured = board[m.to]; u.captured_sq = m.to;
        hash ^= ZOB_PIECE[them][u.captured][m.to];
        remove_piece(them, u.captured, m.to);
        halfmove = 0;
    }

    // Move our piece off `from`.
    hash ^= ZOB_PIECE[us][piece][m.from];
    remove_piece(us, piece, m.from);

    // Place it on `to` (promoting if needed).
    int placed = (m.promo != NO_PIECE) ? m.promo : piece;
    hash ^= ZOB_PIECE[us][placed][m.to];
    put_piece(us, placed, m.to);

    // Move the rook for castling.
    if (m.flag == CASTLE_K) {
        int rf = (us == WHITE) ? 7 : 63, rt = (us == WHITE) ? 5 : 61;
        hash ^= ZOB_PIECE[us][ROOK][rf]; remove_piece(us, ROOK, rf);
        hash ^= ZOB_PIECE[us][ROOK][rt]; put_piece(us, ROOK, rt);
    } else if (m.flag == CASTLE_Q) {
        int rf = (us == WHITE) ? 0 : 56, rt = (us == WHITE) ? 3 : 59;
        hash ^= ZOB_PIECE[us][ROOK][rf]; remove_piece(us, ROOK, rf);
        hash ^= ZOB_PIECE[us][ROOK][rt]; put_piece(us, ROOK, rt);
    }

    // Set a new en-passant target on a double pawn push.
    if (m.flag == DOUBLE_PAWN) {
        new_ep = (us == WHITE) ? m.from + 8 : m.from - 8;
        hash ^= ZOB_EP[file_of(new_ep)];
    }

    // Update castling rights: any king/rook move (or capturing a rook on its home
    // square) removes the relevant rights.
    hash ^= ZOB_CASTLE[castling];
    if (piece == KING) castling &= (us == WHITE) ? ~(WK | WQ) : ~(BK | BQ);
    auto touch = [&](int sq) {
        if (sq == 0) castling &= ~WQ;   else if (sq == 7) castling &= ~WK;
        else if (sq == 56) castling &= ~BQ; else if (sq == 63) castling &= ~BK;
    };
    touch(m.from); touch(m.to);   // moving a rook, or capturing one on its corner
    hash ^= ZOB_CASTLE[castling];

    ep_square = new_ep;
    recompute_aggregates();
    stm = them;
    hash ^= ZOB_STM;
    history.push_back(u);
}

void Board::unmake_move() {
    Undo u = history.back();
    history.pop_back();
    const Move& m = u.move;
    stm ^= 1;                  // back to the side that moved
    Color us = stm, them = stm ^ 1;

    int placed = (m.promo != NO_PIECE) ? m.promo : board[m.to];
    // Remove our piece from `to` (it might be a promoted piece).
    remove_piece(us, placed, m.to);
    int original = (m.promo != NO_PIECE) ? PAWN : placed;
    put_piece(us, original, m.from);

    // Undo rook movement for castling.
    if (m.flag == CASTLE_K) {
        int rf = (us == WHITE) ? 7 : 63, rt = (us == WHITE) ? 5 : 61;
        remove_piece(us, ROOK, rt); put_piece(us, ROOK, rf);
    } else if (m.flag == CASTLE_Q) {
        int rf = (us == WHITE) ? 0 : 56, rt = (us == WHITE) ? 3 : 59;
        remove_piece(us, ROOK, rt); put_piece(us, ROOK, rf);
    }

    // Restore any captured piece.
    if (u.captured != NO_PIECE) put_piece(them, u.captured, u.captured_sq);

    castling = u.castling;
    ep_square = u.ep_square;
    halfmove = u.halfmove;
    hash = u.hash;
    recompute_aggregates();
}

void Board::make_null() {
    // Save just enough to undo: en-passant square and hash (nothing else changes).
    Undo u;
    u.move = {0, 0, NO_PIECE, NORMAL};
    u.captured = NO_PIECE; u.captured_sq = -1;
    u.castling = castling; u.ep_square = ep_square; u.halfmove = halfmove; u.hash = hash;
    history.push_back(u);
    // Clear the en-passant target (a pass can't be answered by en passant) and
    // flip the side to move, keeping the hash consistent.
    if (ep_square >= 0) hash ^= ZOB_EP[file_of(ep_square)];
    ep_square = -1;
    stm ^= 1;
    hash ^= ZOB_STM;
}

void Board::unmake_null() {
    Undo u = history.back();
    history.pop_back();
    stm ^= 1;
    ep_square = u.ep_square;
    castling = u.castling;
    halfmove = u.halfmove;
    hash = u.hash;
}

std::string Board::fen() const {
    // Minimal FEN (placement + stm + castling + ep) — enough for our needs.
    std::string s;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            int sq = r * 8 + f;
            if (board[sq] == NO_PIECE) { empty++; continue; }
            if (empty) { s += char('0' + empty); empty = 0; }
            const char* P = "pnbrqk";
            char c = P[board[sq]];
            s += (color_on[sq] == WHITE) ? char(c - 32) : c;
        }
        if (empty) s += char('0' + empty);
        if (r) s += '/';
    }
    s += (stm == WHITE) ? " w " : " b ";
    return s;
}

}  // namespace chess
