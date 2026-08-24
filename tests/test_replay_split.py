import random

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.replay.split import ReplaySplit, partition_games, split_for_game
from harbichess.search.mcts import MoveStatistics, SearchResult
from harbichess.selfplay.game import SelfPlayConfig, play_game


class OnePlySearch:
    def search(self, state, *, rng: random.Random, add_root_noise: bool):
        del state, rng, add_root_noise
        move = ChessMove("e2e4")
        return SearchResult((MoveStatistics(move, 1, 1.0, 0.0),), 0.0, 1)


def test_game_split_is_deterministic_and_never_splits_positions() -> None:
    assert split_for_game("run-42", validation_fraction=0.2) == split_for_game(
        "run-42", validation_fraction=0.2
    )
    rules = PythonChessRules()
    games = tuple(
        play_game(
            OnePlySearch(),
            rules,
            rules.initial_state(),
            game_index=index,
            seed=index,
            config=SelfPlayConfig(max_plies=1),
        )
        for index in range(20)
    )
    partitions = partition_games(games, run_id="run", validation_fraction=0.25)

    train_ids = {game.game_index for game in partitions[ReplaySplit.TRAIN]}
    validation_ids = {game.game_index for game in partitions[ReplaySplit.VALIDATION]}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == set(range(20))


def test_split_validates_fraction() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        split_for_game("game", validation_fraction=1.0)
