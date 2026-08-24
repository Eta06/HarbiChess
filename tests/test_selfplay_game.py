import random

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, TerminalResult
from harbichess.search.mcts import MoveStatistics, SearchResult
from harbichess.selfplay.game import (
    SelfPlayConfig,
    derive_game_seed,
    play_game,
    play_parallel_games,
)


class ScriptedSearch:
    moves = (ChessMove("f2f3"), ChessMove("e7e5"), ChessMove("g2g4"), ChessMove("d8h4"))

    def search(self, state, *, rng: random.Random, add_root_noise: bool):
        del rng, add_root_noise
        move = self.moves[state.ply]
        return SearchResult((MoveStatistics(move, 1, 1.0, 0.0),), 0.0, 1)


def test_self_play_game_records_policy_and_final_values() -> None:
    rules = PythonChessRules()
    game = play_game(
        ScriptedSearch(),
        rules,
        rules.initial_state(),
        game_index=9,
        seed=derive_game_seed(42, 9),
    )

    assert game.outcome.result is TerminalResult.BLACK_WIN
    assert game.final_state.ply == 4
    assert [sample.selected_move for sample in game.samples] == list(ScriptedSearch.moves)
    assert [sample.outcome_value for sample in game.samples] == [-1, 1, -1, 1]
    assert all(sum(value for _, value in sample.visit_policy) == 1 for sample in game.samples)


def test_parallel_games_receive_reproducible_unique_seeds() -> None:
    rules = PythonChessRules()
    games = play_parallel_games(
        ScriptedSearch(),
        rules,
        run_seed=1234,
        first_game_index=10,
        game_count=4,
        max_workers=4,
    )

    assert [game.game_index for game in games] == [10, 11, 12, 13]
    assert len({game.seed for game in games}) == 4
    assert games[0].seed == derive_game_seed(1234, 10)


def test_self_play_configuration_validation_and_ply_adjudication() -> None:
    with pytest.raises(ValueError, match="ply limits"):
        SelfPlayConfig(max_plies=0)
    with pytest.raises(ValueError, match="game_index"):
        derive_game_seed(1, -1)

    rules = PythonChessRules()
    game = play_game(
        ScriptedSearch(),
        rules,
        rules.initial_state(),
        game_index=0,
        seed=1,
        config=SelfPlayConfig(max_plies=1),
    )
    assert game.outcome.termination == "max_plies"
    assert game.samples[0].outcome_value == 0
