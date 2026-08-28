import random
from types import SimpleNamespace

import pytest

from harbichess.core.state import ChessMove
from harbichess.search.sequential_halving import deterministic_sequential_halving


def _mcts():
    moves = tuple(ChessMove(f"a{index}a{index + 1}") for index in range(1, 8))
    priors = tuple((move, (8 - index) / 28) for index, move in enumerate(moves, 1))
    evaluator = SimpleNamespace(evaluate=lambda _state: SimpleNamespace(priors=priors))
    return SimpleNamespace(evaluator=evaluator)


def test_sequential_halving_spends_exact_budget_and_keeps_best_arm(monkeypatch) -> None:
    values = {f"a{index}a{index + 1}": index / 10 for index in range(1, 8)}
    monkeypatch.setattr(
        "harbichess.search.sequential_halving._continuation_value",
        lambda _mcts, _state, move, *, slots, rng: values[move.uci],
    )

    result = deterministic_sequential_halving(
        _mcts(), object(), budget=64, rng=random.Random(1), maximum_considered_actions=4
    )

    assert result.evaluation_slots == 64
    assert result.rounds == 2
    assert result.selected_action == ChessMove("a4a5")
    assert len(result.considered_actions) == 4
    assert sum(slots for _, slots in result.action_slots) == 63


def test_sequential_halving_rejects_impossible_schedule(monkeypatch) -> None:
    monkeypatch.setattr(
        "harbichess.search.sequential_halving._continuation_value",
        lambda _mcts, _state, move, *, slots, rng: 0.0,
    )
    with pytest.raises(ValueError, match="too small"):
        deterministic_sequential_halving(
            _mcts(), object(), budget=3, rng=random.Random(2), maximum_considered_actions=7
        )


def test_sequential_halving_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="budget"):
        deterministic_sequential_halving(
            _mcts(), object(), budget=2, rng=random.Random(3), maximum_considered_actions=4
        )
