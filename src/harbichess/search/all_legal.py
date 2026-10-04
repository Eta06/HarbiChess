"""Experiment-only, completed-iteration all-legal alpha-beta on full histories."""

import math
import time
from dataclasses import dataclass

import chess

from harbichess.search.tactical_leaf import MaterialQuiescence


class SearchLimit(Exception):
    pass


@dataclass(frozen=True)
class AllLegalResult:
    move: chess.Move | None
    value: float | None
    completed_depth: int
    root_moves_at_completed_depth: int
    nodes: int
    qnodes: int
    evaluations: int
    qdepth_truncations: int
    abort_reason: str | None
    wall_seconds: float
    requested_wall_seconds: float


class AllLegalSearch:
    """No TT, root policy pruning, partial-root publication or checked stand pat."""

    def __init__(self, evaluator, *, max_depth=6, qdepth=6, node_limit=100000):
        if any(type(v) is not int or v <= 0 for v in (max_depth, qdepth, node_limit)):
            raise ValueError("positive integer search limits required")
        self.evaluator = evaluator
        self.max_depth, self.qdepth, self.node_limit = max_depth, qdepth, node_limit

    def search(self, board: chess.Board, *, wall_seconds: float) -> AllLegalResult:
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError("finite positive wall budget required")
        started = time.perf_counter()
        self._deadline = started + wall_seconds
        self._nodes = self._qnodes = self._evaluations = self._qtruncations = 0
        working = board.copy(stack=True)
        outcome = self._terminal(working)
        best_move, best_value, completed, covered, reason = None, outcome, 0, 0, None
        if outcome is None:
            legal = sorted(working.legal_moves, key=lambda m: MaterialQuiescence._order(working, m))
            best_move = legal[0]
            for depth in range(1, self.max_depth + 1):
                try:
                    self._tick(False)
                    order = [best_move] + [m for m in legal if m != best_move]
                    candidate_move, candidate_value, alpha = order[0], -math.inf, -1.0
                    for move in order:
                        working.push(move)
                        try:
                            value = -self._visit(working, depth - 1, -1.0, -alpha)
                        finally:
                            working.pop()
                        if value > candidate_value:
                            candidate_move, candidate_value = move, value
                        alpha = max(alpha, value)
                    # Whole root complete even when an individual subtree had a valid cutoff.
                    best_move, best_value = candidate_move, candidate_value
                    completed, covered = depth, len(order)
                except SearchLimit as error:
                    reason = str(error)
                    break
        return AllLegalResult(
            best_move, best_value, completed, covered, self._nodes, self._qnodes,
            self._evaluations, self._qtruncations, reason, time.perf_counter() - started,
            wall_seconds,
        )

    @staticmethod
    def _terminal(board):
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return None
        return 0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)

    def _tick(self, quiescent):
        if time.perf_counter() >= self._deadline:
            raise SearchLimit("wall")
        if self._nodes + self._qnodes >= self.node_limit:
            raise SearchLimit("nodes")
        if quiescent:
            self._qnodes += 1
        else:
            self._nodes += 1

    def _value(self, board):
        if time.perf_counter() >= self._deadline:
            raise SearchLimit("wall")
        self._evaluations += 1
        value = float(self.evaluator(board))
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("evaluation must be finite STM value in [-1,1]")
        return value

    def _visit(self, board, depth, alpha, beta):
        if depth == 0:
            return self._quiescent(board, self.qdepth, alpha, beta)
        self._tick(False)
        terminal = self._terminal(board)
        if terminal is not None:
            return terminal
        best = -1.0
        for move in sorted(board.legal_moves, key=lambda m: MaterialQuiescence._order(board, m)):
            board.push(move)
            try:
                value = -self._visit(board, depth - 1, -beta, -alpha)
            finally:
                board.pop()
            best, alpha = max(best, value), max(alpha, value)
            if alpha >= beta:
                break
        return best

    def _quiescent(self, board, depth, alpha, beta):
        self._tick(True)
        terminal = self._terminal(board)
        if terminal is not None:
            return terminal
        checked = board.is_check()
        if depth == 0 and checked:
            raise SearchLimit("checked-qdepth")
        best = -1.0 if checked else self._value(board)
        if depth == 0:
            self._qtruncations += 1
            return best
        if not checked:
            if best >= beta:
                return best
            alpha = max(alpha, best)
        moves = sorted(
            (m for m in board.legal_moves if checked or board.is_capture(m) or m.promotion),
            key=lambda m: MaterialQuiescence._order(board, m),
        )
        for move in moves:
            board.push(move)
            try:
                value = -self._quiescent(board, depth - 1, -beta, -alpha)
            finally:
                board.pop()
            best, alpha = max(best, value), max(alpha, value)
            if alpha >= beta:
                break
        return best
