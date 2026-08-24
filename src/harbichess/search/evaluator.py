"""Translate neural policy/WDL outputs into legal search evaluations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from harbichess.chess.actions import POLICY_SIZE, move_to_action
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.backend import (
    EncodedPosition,
    MaskedPolicyValueOutput,
    PolicyValueOutput,
)
from harbichess.core.state import ChessMove, ChessState


@dataclass(frozen=True, slots=True)
class PositionEvaluation:
    priors: tuple[tuple[ChessMove, float], ...]
    value: float


class SearchEvaluator(Protocol):
    def evaluate(self, state: ChessState) -> PositionEvaluation: ...


class EncodedEvaluator(Protocol):
    def evaluate(self, position: EncodedPosition) -> PolicyValueOutput: ...


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("softmax requires at least one value")
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


class NeuralPositionEvaluator:
    """Apply legal masking and return value from the side-to-move perspective."""

    def __init__(
        self,
        evaluator: EncodedEvaluator,
        *,
        rules: PythonChessRules | None = None,
        encoder: BoardEncoder | None = None,
    ) -> None:
        self.rules = rules or PythonChessRules()
        self.encoder = encoder or BoardEncoder(self.rules)
        self.evaluator = evaluator

    def evaluate(self, state: ChessState) -> PositionEvaluation:
        board = self.rules.inspect(state)
        legal_moves = tuple(sorted(board.legal_moves, key=lambda move: move.uci()))
        if not legal_moves:
            raise ValueError("terminal positions must be resolved before neural evaluation")
        encoded = self.encoder.encode_state(state, board)
        actions = tuple(move_to_action(board, move) for move in legal_moves)
        masked_evaluate = getattr(self.evaluator, "evaluate_masked", None)
        if masked_evaluate is None:
            output: PolicyValueOutput | MaskedPolicyValueOutput = self.evaluator.evaluate(encoded)
            if len(output.policy_logits) != POLICY_SIZE:
                raise ValueError(
                    f"policy output must contain {POLICY_SIZE} logits, "
                    f"got {len(output.policy_logits)}"
                )
            legal_logits = tuple(output.policy_logits[action] for action in actions)
        else:
            output = masked_evaluate(encoded, actions)
            if len(output.policy_logits) != len(legal_moves):
                raise ValueError("masked policy output must match the legal action count")
            legal_logits = output.policy_logits
        priors = _softmax(legal_logits)
        win, draw, loss = _softmax(output.wdl_logits)
        del draw
        return PositionEvaluation(
            priors=tuple(
                (ChessMove(move.uci()), prior)
                for move, prior in zip(legal_moves, priors, strict=True)
            ),
            value=win - loss,
        )
