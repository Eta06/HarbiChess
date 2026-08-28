from __future__ import annotations

import random

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.full_gumbel import (
    FullGumbelConfig,
    FullGumbelMCTS,
    considered_visit_sequence,
)


class UniformEvaluator:
    def __init__(self, rules: PythonChessRules, value: float = 0.0) -> None:
        self.rules = rules
        self.value = value

    def evaluate(self, state):
        moves = self.rules.legal_moves(state)
        return PositionEvaluation(
            tuple((move, 1.0 / len(moves)) for move in moves), self.value
        )


def test_considered_visit_sequence_matches_sequential_halving_rounds() -> None:
    assert considered_visit_sequence(4, 16) == (
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
    )
    assert considered_visit_sequence(1, 4) == (0, 1, 2, 3)


def test_full_gumbel_is_fixed_budget_normalized_and_clean_deterministic() -> None:
    rules = PythonChessRules()
    search = FullGumbelMCTS(
        UniformEvaluator(rules),
        rules=rules,
        config=FullGumbelConfig(
            simulations=32,
            max_considered_actions=8,
            gumbel_scale=0.0,
        ),
    )

    first = search.search(
        rules.initial_state(), rng=random.Random(3), add_root_noise=False
    )
    second = search.search(
        rules.initial_state(), rng=random.Random(999), add_root_noise=False
    )

    assert first == second
    assert first.simulations == 32
    assert sum(move.visits for move in first.moves) == 32
    assert sum(weight for _, weight in first.action_weights) == pytest.approx(1.0)
    assert first.select_move(temperature=0.0, rng=random.Random(1)) == first.selected_action


def test_full_gumbel_propagates_terminal_mate_value() -> None:
    rules = PythonChessRules()
    state = rules.initial_state("8/8/8/8/8/8/8/k1KQ4 w - - 0 1")
    search = FullGumbelMCTS(
        UniformEvaluator(rules),
        rules=rules,
        config=FullGumbelConfig(simulations=32, max_considered_actions=16),
    )

    result = search.search(state, rng=random.Random(5), add_root_noise=False)

    assert result.select_move(temperature=0.0, rng=random.Random(0)).uci == "d1a4"
    mate = next(move for move in result.moves if move.move.uci == "d1a4")
    assert mate.mean_value == pytest.approx(1.0)


def test_full_gumbel_rejects_noise_and_bad_configuration() -> None:
    rules = PythonChessRules()
    search = FullGumbelMCTS(UniformEvaluator(rules), rules=rules)
    with pytest.raises(ValueError, match="Dirichlet"):
        search.search(
            rules.initial_state(), rng=random.Random(1), add_root_noise=True
        )
    with pytest.raises(ValueError, match="counts"):
        FullGumbelConfig(simulations=0)
