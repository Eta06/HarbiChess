import random

import pytest

from harbichess.core.state import ChessMove
from harbichess.search.mcts import MoveStatistics, SearchResult
from harbichess.search.root_halving import (
    RootHalvingConfig,
    sequential_halving_root,
)


def _initial() -> SearchResult:
    return SearchResult(
        (
            MoveStatistics(ChessMove("e2e4"), 12, 0.35, 0.02),
            MoveStatistics(ChessMove("d2d4"), 10, 0.30, 0.01),
            MoveStatistics(ChessMove("c2c4"), 6, 0.20, 0.00),
            MoveStatistics(ChessMove("g1f3"), 4, 0.15, -0.01),
        ),
        0.01,
        32,
    )


def test_consensus_winner_receives_meaningful_policy_transfer(monkeypatch) -> None:
    values = {
        ("e2e4", 3): 0.24,
        ("d2d4", 3): 0.12,
        ("c2c4", 3): 0.03,
        ("g1f3", 3): -0.02,
        ("e2e4", 7): 0.28,
        ("d2d4", 7): 0.11,
    }
    monkeypatch.setattr(
        "harbichess.search.root_halving._continuation_value",
        lambda _mcts, _state, move, *, simulations, rng: values[(move.uci, simulations)],
    )

    result = sequential_halving_root(
        object(),
        object(),
        _initial(),
        rng=random.Random(1),
    )

    assert result.evidence is not None
    assert result.evidence.adjusted
    assert result.evidence.winner == ChessMove("e2e4")
    assert result.evidence.adjusted_winner_mass > result.evidence.original_winner_mass
    assert result.search.simulations == 64
    assert result.search.moves[0].visits > _initial().moves[0].visits


def test_round_disagreement_preserves_visit_policy(monkeypatch) -> None:
    values = {
        ("e2e4", 3): 0.24,
        ("d2d4", 3): 0.12,
        ("c2c4", 3): 0.03,
        ("g1f3", 3): -0.02,
        ("e2e4", 7): 0.08,
        ("d2d4", 7): 0.20,
    }
    monkeypatch.setattr(
        "harbichess.search.root_halving._continuation_value",
        lambda _mcts, _state, move, *, simulations, rng: values[(move.uci, simulations)],
    )

    result = sequential_halving_root(
        object(),
        object(),
        _initial(),
        rng=random.Random(2),
    )

    assert result.evidence is not None
    assert not result.evidence.adjusted
    assert [move.visits for move in result.search.moves] == [12, 10, 6, 4]


def test_too_few_visited_actions_spends_no_forced_budget() -> None:
    initial = SearchResult(_initial().moves[:3], 0.0, 32)
    result = sequential_halving_root(
        object(),
        object(),
        initial,
        rng=random.Random(3),
    )

    assert result.search is initial
    assert result.evidence is None


def test_root_halving_budget_and_configuration_are_frozen() -> None:
    config = RootHalvingConfig()
    assert config.forced_evaluations == 32
    with pytest.raises(ValueError, match="invalid"):
        RootHalvingConfig(finalists=4)
