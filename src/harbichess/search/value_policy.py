"""Turn MCTS visit counts into a conservative value-improved policy target."""

from __future__ import annotations

import math
from dataclasses import dataclass

from harbichess.core.state import ChessMove
from harbichess.search.mcts import MoveStatistics


@dataclass(frozen=True, slots=True)
class ValueImprovedPolicyConfig:
    """Controls bounded action-value improvement of a visit-count target."""

    advantage_temperature: float = 0.10
    prior_visits: float = 8.0
    maximum_logit_adjustment: float = 1.25

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.advantage_temperature)
            or self.advantage_temperature <= 0
            or not math.isfinite(self.prior_visits)
            or self.prior_visits < 0
            or not math.isfinite(self.maximum_logit_adjustment)
            or self.maximum_logit_adjustment < 0
        ):
            raise ValueError("value-improved policy parameters must be finite and valid")


def value_improved_policy(
    moves: tuple[MoveStatistics, ...],
    root_value: float,
    *,
    config: ValueImprovedPolicyConfig | None = None,
) -> tuple[tuple[ChessMove, float], ...]:
    """Reweight visits by a shrinkage estimate of root-relative action value.

    Low-visit action values are pulled toward the root estimate, while a hard
    logit bound prevents a noisy value estimate from overwhelming search visits.
    """

    settings = config or ValueImprovedPolicyConfig()
    if not math.isfinite(root_value) or not -1.0 <= root_value <= 1.0:
        raise ValueError("root_value must be finite and within [-1, 1]")
    visited = tuple(move for move in moves if move.visits > 0)
    if not visited:
        raise ValueError("value-improved policy requires at least one visited move")
    if any(
        not math.isfinite(move.mean_value) or not -1.0 <= move.mean_value <= 1.0
        for move in visited
    ):
        raise ValueError("move values must be finite and within [-1, 1]")

    logits = []
    for move in visited:
        shrunk_value = (
            move.visits * move.mean_value + settings.prior_visits * root_value
        ) / (move.visits + settings.prior_visits)
        adjustment = (shrunk_value - root_value) / settings.advantage_temperature
        bounded_adjustment = max(
            -settings.maximum_logit_adjustment,
            min(settings.maximum_logit_adjustment, adjustment),
        )
        logits.append(math.log(move.visits) + bounded_adjustment)

    maximum = max(logits)
    weights = tuple(math.exp(logit - maximum) for logit in logits)
    total = sum(weights)
    return tuple(
        (move.move, weight / total)
        for move, weight in zip(visited, weights, strict=True)
    )
