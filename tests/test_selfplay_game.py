import random

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, TerminalResult
from harbichess.search.mcts import MoveStatistics, SearchResult
from harbichess.selfplay.game import (
    SelfPlayConfig,
    _redirect_repetition,
    derive_game_seed,
    play_game,
    play_parallel_games,
)


def _threefold_choice_state(rules: PythonChessRules):
    state = rules.initial_state()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        state = rules.apply(state, ChessMove(move))
    return state


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
    completed = []
    games = play_parallel_games(
        ScriptedSearch(),
        rules,
        run_seed=1234,
        first_game_index=10,
        game_count=4,
        max_workers=4,
        on_game_complete=completed.append,
    )

    assert [game.game_index for game in games] == [10, 11, 12, 13]
    assert len({game.seed for game in games}) == 4
    assert games[0].seed == derive_game_seed(1234, 10)
    assert sorted(game.game_index for game in completed) == [10, 11, 12, 13]


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


def test_repetition_target_redirects_to_comparable_non_repeating_move() -> None:
    rules = PythonChessRules()
    state = _threefold_choice_state(rules)
    repeating = ChessMove("f6g8")
    alternative = ChessMove("h8g8")
    search = SearchResult(
        (
            MoveStatistics(repeating, 12, 0.6, 0.0),
            MoveStatistics(alternative, 4, 0.4, -0.03),
        ),
        0.0,
        16,
    )

    policy_moves, selected, redirected = _redirect_repetition(
        search,
        rules,
        state,
        repeating,
        temperature=0.0,
        tolerance=0.05,
        rng=random.Random(1),
    )

    assert redirected
    assert selected == alternative
    assert tuple(item.move for item in policy_moves) == (alternative,)


def test_repetition_target_keeps_draw_when_continuations_are_worse() -> None:
    rules = PythonChessRules()
    state = _threefold_choice_state(rules)
    repeating = ChessMove("f6g8")
    alternative = ChessMove("h8g8")
    search = SearchResult(
        (
            MoveStatistics(repeating, 12, 0.6, 0.0),
            MoveStatistics(alternative, 4, 0.4, -0.25),
        ),
        0.0,
        16,
    )

    policy_moves, selected, redirected = _redirect_repetition(
        search,
        rules,
        state,
        repeating,
        temperature=0.0,
        tolerance=0.05,
        rng=random.Random(1),
    )

    assert not redirected
    assert selected == repeating
    assert policy_moves == search.moves
