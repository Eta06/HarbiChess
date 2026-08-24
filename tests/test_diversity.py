import random

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.replay.diversity import measure_diversity
from harbichess.search.mcts import MoveStatistics, SearchResult
from harbichess.selfplay.game import SelfPlayConfig, play_game


class FirstMoveSearch:
    def __init__(self, move: str) -> None:
        self.move = ChessMove(move)

    def search(self, state, *, rng: random.Random, add_root_noise: bool):
        del state, rng, add_root_noise
        return SearchResult((MoveStatistics(self.move, 1, 1.0, 0.0),), 0.0, 1)


def one_ply_game(index: int, move: str):
    rules = PythonChessRules()
    return play_game(
        FirstMoveSearch(move),
        rules,
        rules.initial_state(),
        game_index=index,
        seed=index,
        config=SelfPlayConfig(max_plies=1),
    )


def test_diversity_detects_duplicate_games_and_action_coverage() -> None:
    games = (
        one_ply_game(0, "e2e4"),
        one_ply_game(1, "e2e4"),
        one_ply_game(2, "d2d4"),
        one_ply_game(3, "d2d4"),
    )

    metrics = measure_diversity(games, opening_plies=(1, 4))

    assert metrics.unique_game_ratio == 0.5
    assert metrics.duplicate_game_ratio == 0.5
    assert metrics.unique_position_ratio == 0.25
    assert metrics.selected_actions == 2
    assert metrics.effective_policy_branches == 1.0
    assert metrics.draws == 4
    assert metrics.decisive_games == 0
    assert metrics.max_ply_draws == 4
    assert metrics.max_ply_draw_ratio == 1.0
    assert [(item.termination, item.count) for item in metrics.terminations] == [
        ("max_plies", 4)
    ]
    assert metrics.openings[0].unique_prefixes == 2
    assert metrics.openings[0].effective_prefixes == pytest.approx(2.0)
    assert metrics.openings[1].eligible_games == 0


def test_diversity_requires_games_and_positive_depths() -> None:
    with pytest.raises(ValueError, match="requires games"):
        measure_diversity(())
    with pytest.raises(ValueError, match="positive"):
        measure_diversity((one_ply_game(0, "e2e4"),), opening_plies=(0,))
