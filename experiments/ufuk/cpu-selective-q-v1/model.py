"""Cheap sixteen-state-feature search-effort classifier; no learned value."""

import math

import chess

DIM = 16
SCHEMA = "own-selective-quiescence-effort-model-v1"


def features(board, static):
    legal = list(board.legal_moves)
    captures = [m for m in legal if board.is_capture(m)]
    phase = sum(
        len(board.pieces(p, c)) * w
        for p, w in [(2, 1), (3, 1), (4, 2), (5, 4)]
        for c in [True, False]
    )
    king, other = board.king(board.turn), board.king(not board.turn)

    def rank(s):
        return (chess.square_rank(s) if board.turn else 7 - chess.square_rank(s)) / 7

    own = sum(len(board.pieces(p, board.turn)) for p in range(2, 6))
    opp = sum(len(board.pieces(p, not board.turn)) for p in range(2, 6))
    return [
        1.0,
        float(board.is_check()),
        min(phase, 24) / 24,
        min(len(legal), 64) / 64,
        min(len(captures), 32) / 32,
        sum(bool(m.promotion) for m in legal) / 8,
        float(static),
        abs(float(static)),
        min(board.halfmove_clock, 100) / 100,
        min(chess.square_file(king), 7 - chess.square_file(king)) / 3,
        rank(king),
        rank(other),
        min(own, 16) / 16,
        min(opp, 16) / 16,
        max([board.piece_type_at(m.to_square) or 1 for m in captures] + [0]) / 6,
        max([board.piece_type_at(m.from_square) for m in captures] + [0]) / 6,
    ]


def probability(weights, x):
    z = sum(a * b for a, b in zip(weights, x, strict=True))
    return 1 / (1 + math.exp(-max(-60, min(60, z))))


def model_dict(weights):
    if len(weights) != DIM or any(not math.isfinite(x) for x in weights):
        raise ValueError("finite16 weights required")
    return dict(
        schema=SCHEMA,
        dimensions=DIM,
        weights=list(weights),
        threshold=0.5,
        encoding="cheap-mover-fullhistory-state16-v1",
        evaluator="unchanged-humanprior18",
        nominal_qdepth=2,
        extra_qdepth=2,
        extra_node_cap=64,
    )


def load_model(packet):
    if packet != model_dict(packet["weights"]):
        raise ValueError("exact effort schema required")
    return list(packet["weights"])
