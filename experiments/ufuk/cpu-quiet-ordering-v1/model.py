"""268 sparse cheap quiet-action features; no NN/check oracle/value learning."""

import math

import chess

DIM = 268
SCHEMA = "classical-own-quiet-ordering-model-v1"
ENCODING = "mover-relative-filefold-piece-from-to-phase-castle-v1"


def quiet(board, move):
    return not move.promotion and not board.is_capture(move)


def square_index(square, turn):
    rank = chess.square_rank(square) if turn else 7 - chess.square_rank(square)
    file = chess.square_file(square)
    return rank * 4 + min(file, 7 - file)


def features(board, move):
    if move not in board.legal_moves or not quiet(board, move):
        raise ValueError("legal quiet action required")
    piece = board.piece_type_at(move.from_square) - 1
    src, dst = square_index(move.from_square, board.turn), square_index(move.to_square, board.turn)
    phase = sum(
        len(board.pieces(p, c)) * weight
        for p, weight in [(chess.KNIGHT, 1), (chess.BISHOP, 1), (chess.ROOK, 2), (chess.QUEEN, 4)]
        for c in (chess.WHITE, chess.BLACK)
    )
    indices = [piece, 6 + src, 38 + dst, 70 + piece * 32 + dst, 262 + min(3, phase // 7)]
    if board.is_castling(move):
        indices.append(266 + int(chess.square_file(move.to_square) < 4))
    return tuple(indices)


def score(weights, packet):
    return sum(weights[i] for i in packet)


def model_dict(weights):
    if len(weights) != DIM or any(not math.isfinite(x) for x in weights):
        raise ValueError("finite268 weight storage required")
    return dict(
        schema=SCHEMA,
        action_encoding=ENCODING,
        dimensions=DIM,
        weights=list(weights),
        evaluator_identity="unchanged-humanprior18-theta0;ordering-only",
    )


def load_model(obj):
    if obj != model_dict(obj["weights"]):
        raise ValueError("exact ordering model schema/encoding/storage required")
    return list(obj["weights"])
