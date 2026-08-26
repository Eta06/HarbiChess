"""Deterministic tactical/material values for isolated search diagnostics."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import chess

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.search.evaluator import PositionEvaluation, SearchEvaluator

_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


@dataclass(frozen=True, slots=True)
class TacticalOracleConfig:
    depth: int = 2
    material_scale: float = 39.0
    cache_size: int = 100_000

    def __post_init__(self) -> None:
        if self.depth < 0 or self.material_scale <= 0 or self.cache_size <= 0:
            raise ValueError("tactical oracle configuration is invalid")


class DeterministicTacticalOracle:
    """Bounded negamax over checks, captures, promotions, and forced evasions."""

    def __init__(
        self,
        *,
        rules: PythonChessRules,
        config: TacticalOracleConfig | None = None,
    ) -> None:
        self.rules = rules
        self.config = config or TacticalOracleConfig()
        self._thread_local = threading.local()

    def value(self, state: ChessState, *, depth: int | None = None) -> float:
        remaining = self.config.depth if depth is None else depth
        if remaining < 0:
            raise ValueError("oracle depth must be non-negative")
        return self._negamax(state, remaining)

    def _cache(self) -> dict[tuple[ChessState, int], float]:
        cache = getattr(self._thread_local, "cache", None)
        if cache is None:
            cache = {}
            self._thread_local.cache = cache
        return cache

    def _negamax(self, state: ChessState, depth: int) -> float:
        key = (state, depth)
        cache = self._cache()
        cached = cache.get(key)
        if cached is not None:
            return cached
        outcome = self.rules.outcome(state, claim_draw=True)
        if outcome is not None:
            value = float(outcome.value_for(self.rules.view(state).side_to_move))
        else:
            board = self.rules.inspect(state)
            stand_pat = self._material_value(board)
            if depth == 0:
                value = stand_pat
            else:
                moves = self._tactical_moves(board)
                if not moves:
                    value = stand_pat
                else:
                    continuations = tuple(
                        -self._negamax(self.rules.apply(state, move), depth - 1)
                        for move in moves
                    )
                    value = max(continuations) if board.is_check() else max(
                        stand_pat, *continuations
                    )
        if len(cache) >= self.config.cache_size:
            cache.clear()
        cache[key] = value
        return value

    def _material_value(self, board: chess.Board) -> float:
        own = sum(
            len(board.pieces(piece_type, board.turn)) * piece_value
            for piece_type, piece_value in _PIECE_VALUES.items()
        )
        opponent = sum(
            len(board.pieces(piece_type, not board.turn)) * piece_value
            for piece_type, piece_value in _PIECE_VALUES.items()
        )
        return math.tanh((own - opponent) / self.config.material_scale)

    @staticmethod
    def _tactical_moves(board: chess.Board) -> tuple[ChessMove, ...]:
        moves = tuple(sorted(board.legal_moves, key=lambda move: move.uci()))
        if board.is_check():
            return tuple(ChessMove(move.uci()) for move in moves)
        return tuple(
            ChessMove(move.uci())
            for move in moves
            if board.is_capture(move) or move.promotion is not None or board.gives_check(move)
        )


class OracleValueEvaluator:
    """Keep neural policy priors unchanged and replace only the leaf value."""

    def __init__(
        self,
        policy_evaluator: SearchEvaluator,
        oracle: DeterministicTacticalOracle,
    ) -> None:
        self.policy_evaluator = policy_evaluator
        self.oracle = oracle

    def evaluate(self, state: ChessState) -> PositionEvaluation:
        neural = self.policy_evaluator.evaluate(state)
        return PositionEvaluation(neural.priors, self.oracle.value(state))
