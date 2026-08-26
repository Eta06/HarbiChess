import math

import pytest

from harbichess.core.state import ChessMove
from harbichess.search.mcts import MoveStatistics
from harbichess.search.value_policy import (
    ValueImprovedPolicyConfig,
    value_improved_policy,
)


def _moves(*items: tuple[str, int, float]):
    return tuple(
        MoveStatistics(ChessMove(move), visits, 0.5, value)
        for move, visits, value in items
    )


def test_equal_action_values_preserve_visit_distribution() -> None:
    policy = value_improved_policy(
        _moves(("e2e4", 24, 0.2), ("d2d4", 8, 0.2)),
        0.2,
    )

    assert dict(policy) == pytest.approx(
        {ChessMove("e2e4"): 0.75, ChessMove("d2d4"): 0.25}
    )


def test_reliable_positive_advantage_gains_policy_mass() -> None:
    raw_mass = 0.5
    policy = dict(
        value_improved_policy(
            _moves(("e2e4", 16, 0.25), ("d2d4", 16, -0.05)),
            0.0,
        )
    )

    assert policy[ChessMove("e2e4")] > raw_mass
    assert sum(policy.values()) == pytest.approx(1.0)


def test_low_visit_value_is_shrunk_and_logit_adjustment_is_bounded() -> None:
    config = ValueImprovedPolicyConfig(
        advantage_temperature=0.01,
        prior_visits=8,
        maximum_logit_adjustment=0.4,
    )
    policy = dict(
        value_improved_policy(
            _moves(("e2e4", 1, 1.0), ("d2d4", 1, -1.0)),
            0.0,
            config=config,
        )
    )

    odds = policy[ChessMove("e2e4")] / policy[ChessMove("d2d4")]
    assert odds == pytest.approx(math.exp(0.8))


def test_value_improved_policy_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="visited move"):
        value_improved_policy(_moves(("e2e4", 0, 0.0)), 0.0)
    with pytest.raises(ValueError, match="root_value"):
        value_improved_policy(_moves(("e2e4", 1, 0.0)), float("nan"))
    with pytest.raises(ValueError, match="finite and valid"):
        ValueImprovedPolicyConfig(advantage_temperature=0.0)
