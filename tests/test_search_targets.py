import pytest

from harbichess.core.state import ChessMove
from harbichess.search.mcts import MoveStatistics, SearchResult
from harbichess.search.targets import prune_noise_attributable_visits, visit_policy


def test_noise_target_pruning_preserves_leader_and_removes_boosted_visits() -> None:
    leader = ChessMove("e2e4")
    boosted = ChessMove("a2a3")
    search = SearchResult(
        (
            MoveStatistics(leader, 10, 0.45, 0.1),
            MoveStatistics(boosted, 6, 0.40, 0.0),
        ),
        0.05,
        16,
    )

    raw = dict(visit_policy(search))
    pruned = dict(prune_noise_attributable_visits(search, {leader: 0.55, boosted: 0.05}))

    assert sum(pruned.values()) == pytest.approx(1.0)
    assert pruned[leader] > raw[leader]
    assert pruned[boosted] < raw[boosted]


def test_noise_target_pruning_requires_complete_clean_priors() -> None:
    move = ChessMove("e2e4")
    search = SearchResult((MoveStatistics(move, 1, 1.0, 0.0),), 0.0, 1)

    with pytest.raises(ValueError, match="cover"):
        prune_noise_attributable_visits(search, {})
