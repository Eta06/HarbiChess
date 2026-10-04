"""Experiment-only classical leaf controls; neural policy priors stay unchanged."""

import math
from dataclasses import dataclass

import chess

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessState
from harbichess.search.evaluator import PositionEvaluation, SearchEvaluator

PIECE_UNITS = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def material_value(board: chess.Board) -> float:
    white = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_UNITS.items()
    )
    return 0.99 * math.tanh((white if board.turn else -white) / 600)


@dataclass
class TacticalCounters:
    calls: int = 0
    nodes: int = 0
    node_truncations: int = 0
    depth_truncations: int = 0
    checked_fallback_calls: int = 0


class MaterialQuiescence:
    """Bounded approximate value, never a claim of complete minimax bounds."""

    def __init__(self, *, depth=4, node_limit=64):
        if any(type(v) is not int or v <= 0 for v in (depth, node_limit)):
            raise ValueError("positive integer tactical depth and node limit required")
        self.depth, self.node_limit = depth, node_limit
        self.counters = TacticalCounters()
        self.last_nodes = 0
        self._checked_fallback = False

    def evaluate(self, board: chess.Board) -> float:
        self.last_nodes = 0
        self._checked_fallback = False
        self.counters.calls += 1
        value = self._visit(board, self.depth, -1.0, 1.0)
        self.counters.checked_fallback_calls += int(self._checked_fallback)
        return value

    @staticmethod
    def _order(board, move):
        victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
        gain = PIECE_UNITS.get(victim, 0) + PIECE_UNITS.get(move.promotion, 0)
        cost = PIECE_UNITS.get(board.piece_type_at(move.from_square), 0)
        return (-gain, cost, move.uci())

    def _visit(self, board, depth, alpha, beta):
        self.last_nodes += 1
        self.counters.nodes += 1
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return (
                0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)
            )
        checked = board.is_check()
        if depth == 0:
            self.counters.depth_truncations += 1
            self._checked_fallback |= checked
            return material_value(board)
        best = -1.0 if checked else material_value(board)
        if not checked:
            if best >= beta:
                return best
            alpha = max(alpha, best)
        moves = sorted(
            (
                move
                for move in board.legal_moves
                if checked or board.is_capture(move) or move.promotion
            ),
            key=lambda move: self._order(board, move),
        )
        for move_index, move in enumerate(moves):
            if self.last_nodes >= self.node_limit:
                self.counters.node_truncations += 1
                if checked and move_index == 0:
                    self._checked_fallback = True
                    return material_value(board)
                break
            board.push(move)
            try:
                value = -self._visit(board, depth - 1, -beta, -alpha)
            finally:
                board.pop()
            best = max(best, value)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best


class TacticalLeafEvaluator:
    def __init__(
        self,
        evaluator: SearchEvaluator,
        *,
        mode="quiescent",
        rules: PythonChessRules | None = None,
        depth=4,
        node_limit=64,
    ):
        if mode not in ("static", "quiescent"):
            raise ValueError("tactical leaf mode must be static or quiescent")
        self.evaluator, self.mode = evaluator, mode
        self.rules = rules or PythonChessRules()
        self.tactical = MaterialQuiescence(depth=depth, node_limit=node_limit)

    def evaluate(self, state: ChessState) -> PositionEvaluation:
        evaluation = self.evaluator.evaluate(state)
        board = self.rules.board(state)
        value = material_value(board) if self.mode == "static" else self.tactical.evaluate(board)
        return PositionEvaluation(evaluation.priors, value)
