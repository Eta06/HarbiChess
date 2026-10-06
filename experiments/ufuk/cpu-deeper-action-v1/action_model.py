"""All-legal sparse280 action code; state18 archived, not cancelling CE biases."""

import math

import chess

DIM = 280
SCHEMA = "own-deeper-search-bestmove-model-v1"
EPSILON = 0.01


def features(board, move):
    if move not in board.legal_moves:
        raise ValueError("legal action required")
    piece = board.piece_type_at(move.from_square) - 1

    def square(sq):
        return (chess.square_rank(sq) if board.turn else 7 - chess.square_rank(sq)) * 4 + min(
            chess.square_file(sq), 7 - chess.square_file(sq)
        )

    src, dst = square(move.from_square), square(move.to_square)
    phase = sum(
        len(board.pieces(p, c)) * w
        for p, w in [(2, 1), (3, 1), (4, 2), (5, 4)]
        for c in (True, False)
    )
    packet = [piece, 6 + src, 38 + dst, 70 + piece * 32 + dst, 262 + min(3, phase // 7)]
    if board.is_castling(move):
        packet.append(266 + int(chess.square_file(move.to_square) < 4))
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square) or int(board.is_en_passant(move))
        packet.append(268 + victim - 1)
    if move.promotion:
        packet.append(274 + move.promotion - 1)
    return tuple(packet)


def score(weights, packet):
    return sum(weights[i] for i in packet)


def probabilities(weights, packets):
    logits = [score(weights, p) for p in packets]
    maximum = max(logits)
    values = [math.exp(x - maximum) for x in logits]
    return [x / sum(values) for x in values]


def model_dict(weights):
    if len(weights) != DIM or any(not math.isfinite(x) for x in weights):
        raise ValueError("finite280 weights required")
    return dict(
        schema=SCHEMA,
        dimensions=DIM,
        weights=list(weights),
        encoding="mover-filefold-piece-from-to-phase-castle-victim-promotion280-v1",
        root_epsilon=EPSILON,
        label="selected-bestmove-only-no-visits",
    )


def load_model(obj):
    if obj != model_dict(obj["weights"]):
        raise ValueError("exact model schema")
    return list(obj["weights"])
