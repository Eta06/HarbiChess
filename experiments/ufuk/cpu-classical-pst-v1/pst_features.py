"""Sparse color/rank/file-symmetric piece-square residual features."""

import chess

TABLE_WIDTH = 32
NONKING_TABLES = 5
KING_TABLES = 2
FEATURE_COUNT = TABLE_WIDTH * (NONKING_TABLES + KING_TABLES)


def material_phase(board):
    weights = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    material = sum(
        weights[piece] * board.pieces_mask(piece, color).bit_count()
        for piece in weights
        for color in (chess.WHITE, chess.BLACK)
    )
    return min(24, material) / 24.0


def piece_square_features(board):
    """Mover-relative piece counts on 4x8 mirrored squares plus phase-split kings."""
    phase = material_phase(board)
    features = [0.0] * FEATURE_COUNT
    mover = board.turn
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == mover else -1.0
        for piece in range(chess.PAWN, chess.KING):
            bits = board.pieces_mask(piece, color)
            offset = (piece - 1) * TABLE_WIDTH
            while bits:
                bit = bits & -bits
                bits ^= bit
                square = bit.bit_length() - 1
                oriented = square if color == chess.WHITE else square ^ 56
                rank, file = divmod(oriented, 8)
                index = rank * 4 + min(file, 7 - file)
                features[offset + index] += sign
        king = board.king(color)
        if king is not None:
            oriented = king if color == chess.WHITE else king ^ 56
            rank, file = divmod(oriented, 8)
            index = rank * 4 + min(file, 7 - file)
            features[NONKING_TABLES * TABLE_WIDTH + index] += sign * phase
            features[(NONKING_TABLES + 1) * TABLE_WIDTH + index] += sign * (1.0 - phase)
    return tuple(features)
