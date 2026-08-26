import random

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.gumbel import GumbelSearchConfig, gumbel_sequential_halving
from harbichess.search.mcts import MCTS, SearchConfig


class UniformEvaluator:
    def __init__(self, rules: PythonChessRules) -> None:
        self.rules = rules

    def evaluate(self, state):
        moves = self.rules.legal_moves(state)
        return PositionEvaluation(tuple((move, 1.0 / len(moves)) for move in moves), 0.0)


def test_gumbel_sequential_halving_is_fixed_budget_and_reproducible() -> None:
    rules = PythonChessRules()
    search = MCTS(
        UniformEvaluator(rules),
        rules=rules,
        config=SearchConfig(simulations=1, dirichlet_fraction=0.0),
    )
    config = GumbelSearchConfig(simulations=16, top_actions=4)

    first = gumbel_sequential_halving(
        search,
        rules.initial_state(),
        rng=random.Random(17),
        config=config,
    )
    second = gumbel_sequential_halving(
        search,
        rules.initial_state(),
        rng=random.Random(17),
        config=config,
    )

    assert first == second
    assert first.simulations == 16
    assert len(first.sampled_actions) == 4
    assert first.selected_move in first.sampled_actions
    assert sum(probability for _, probability in first.policy) == pytest.approx(1.0)


def test_gumbel_search_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="configuration"):
        GumbelSearchConfig(simulations=0)
