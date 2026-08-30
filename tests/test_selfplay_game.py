import math
import random

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, TerminalResult
from harbichess.search.continuation import transform_repetition_target
from harbichess.search.full_gumbel import FullGumbelSearchResult
from harbichess.search.mcts import MoveStatistics, SearchResult
from harbichess.selfplay.game import (
    SelfPlayConfig,
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


def test_self_play_records_clean_prior_to_teacher_policy_telemetry() -> None:
    rules = PythonChessRules()
    first = ChessMove("e2e4")
    second = ChessMove("d2d4")

    class TeacherSearch:
        def search(self, state, *, rng: random.Random, add_root_noise: bool):
            del state, rng, add_root_noise
            return SearchResult(
                (
                    MoveStatistics(first, 2, 0.7, 0.10),
                    MoveStatistics(second, 8, 0.3, 0.35),
                ),
                0.2,
                10,
                network_priors=((first, 0.7), (second, 0.3)),
            )

    game = play_game(
        TeacherSearch(),
        rules,
        rules.initial_state(),
        game_index=0,
        seed=1,
        config=SelfPlayConfig(max_plies=1, temperature=0.0),
    )
    sample = game.samples[0]

    assert sample.raw_policy == ((second, 0.3), (first, 0.7))
    assert sample.teacher_argmax_changed
    assert sample.teacher_policy_tv == pytest.approx(0.5)
    assert sample.teacher_policy_kl == pytest.approx(
        0.2 * math.log(0.2 / 0.7) + 0.8 * math.log(0.8 / 0.3)
    )
    assert sample.teacher_search_value_delta == pytest.approx(0.25)


def test_full_gumbel_replay_uses_improved_policy_instead_of_sparse_visits() -> None:
    rules = PythonChessRules()
    visited = ChessMove("e2e4")
    selected = ChessMove("d2d4")

    class GumbelSearch:
        def search(self, state, *, rng: random.Random, add_root_noise: bool):
            del state, rng, add_root_noise
            return FullGumbelSearchResult(
                moves=(
                    MoveStatistics(visited, 4, 0.6, 0.1),
                    MoveStatistics(selected, 0, 0.4, 0.2),
                ),
                root_value=0.1,
                simulations=4,
                selected_action=selected,
                action_weights=((visited, 0.45), (selected, 0.55)),
            )

    game = play_game(
        GumbelSearch(),
        rules,
        rules.initial_state(),
        game_index=0,
        seed=1,
        config=SelfPlayConfig(max_plies=1, temperature=0.0),
    )

    assert game.samples[0].selected_move == selected
    assert game.samples[0].visit_policy == ((visited, 0.45), (selected, 0.55))


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


def test_parallel_games_preserve_supplied_continuation_states() -> None:
    rules = PythonChessRules()
    first = rules.apply(rules.initial_state(), ChessMove("f2f3"))
    second = rules.apply(rules.initial_state(), ChessMove("e2e4"))

    games = play_parallel_games(
        ScriptedSearch(),
        rules,
        run_seed=42,
        first_game_index=0,
        game_count=2,
        max_workers=2,
        config=SelfPlayConfig(max_plies=1),
        initial_states=(first, second),
        max_additional_plies=1,
    )

    assert tuple(game.samples[0].state for game in games) == (first, second)
    assert tuple(game.final_state.ply for game in games) == (2, 2)


def test_parallel_games_require_one_continuation_state_per_game() -> None:
    rules = PythonChessRules()

    with pytest.raises(ValueError, match="initial state count"):
        play_parallel_games(
            ScriptedSearch(),
            rules,
            run_seed=42,
            first_game_index=0,
            game_count=2,
            max_workers=2,
            initial_states=(rules.initial_state(),),
        )
    with pytest.raises(ValueError, match="additional ply limit"):
        play_parallel_games(
            ScriptedSearch(),
            rules,
            run_seed=42,
            first_game_index=0,
            game_count=1,
            max_workers=1,
            max_additional_plies=0,
        )


def test_self_play_configuration_validation_and_ply_adjudication() -> None:
    with pytest.raises(ValueError, match="ply limits"):
        SelfPlayConfig(max_plies=0)
    with pytest.raises(ValueError, match="game_index"):
        derive_game_seed(1, -1)
    with pytest.raises(ValueError, match="temperatures"):
        SelfPlayConfig(selection_dirichlet_fraction=0.25)

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
    assert game.samples[0].outcome_value is None


def test_selection_noise_keeps_search_teacher_clean() -> None:
    rules = PythonChessRules()
    first = ChessMove("e2e4")
    second = ChessMove("d2d4")
    observed_noise_flags = []

    class CleanTeacherSearch:
        def search(self, state, *, rng: random.Random, add_root_noise: bool):
            del state, rng
            observed_noise_flags.append(add_root_noise)
            return SearchResult(
                (
                    MoveStatistics(first, 9, 0.7, 0.2),
                    MoveStatistics(second, 1, 0.3, 0.1),
                ),
                0.1,
                10,
                network_priors=((first, 0.7), (second, 0.3)),
            )

    game = play_game(
        CleanTeacherSearch(),
        rules,
        rules.initial_state(),
        game_index=0,
        seed=5,
        config=SelfPlayConfig(
            max_plies=1,
            search_root_noise=False,
            selection_dirichlet_fraction=0.25,
        ),
    )

    assert observed_noise_flags == [False]
    assert game.samples[0].visit_policy == ((first, 0.9), (second, 0.1))


def test_dual_search_decouples_noisy_behavior_from_clean_target() -> None:
    rules = PythonChessRules()
    clean = ChessMove("e2e4")
    behavior = ChessMove("d2d4")

    class DualSearch:
        def search(self, state, *, rng: random.Random, add_root_noise: bool):
            del state, rng
            selected = behavior if add_root_noise else clean
            return SearchResult(
                (MoveStatistics(selected, 8, 1.0, 0.2),),
                0.2,
                8,
                network_priors=((clean, 0.5), (behavior, 0.5)),
            )

    game = play_game(
        DualSearch(),
        rules,
        rules.initial_state(),
        game_index=0,
        seed=7,
        config=SelfPlayConfig(max_plies=1, separate_clean_target_search=True),
    )
    sample = game.samples[0]

    assert sample.selected_move == behavior
    assert sample.visit_policy == ((clean, 1.0),)
    assert sample.behavior_target_decoupled


def test_repetition_target_transform_is_opt_in_for_self_play() -> None:
    class NonTerminalTestRules(PythonChessRules):
        def outcome(self, state, *, claim_draw: bool = False):
            del state, claim_draw
            return None

    rules = NonTerminalTestRules()
    state = _threefold_choice_state(rules)
    repeating = ChessMove("f6g8")
    alternative = ChessMove("h8g8")

    class RepetitionSearch:
        def search(self, state, *, rng: random.Random, add_root_noise: bool):
            del state, rng, add_root_noise
            return SearchResult(
                (
                    MoveStatistics(repeating, 12, 0.6, 0.0),
                    MoveStatistics(alternative, 4, 0.4, -0.03),
                ),
                0.0,
                16,
            )

    default_game = play_game(
        RepetitionSearch(),
        rules,
        state,
        game_index=0,
        seed=1,
        config=SelfPlayConfig(max_plies=state.ply + 1),
    )
    legacy_game = play_game(
        RepetitionSearch(),
        rules,
        state,
        game_index=1,
        seed=1,
        config=SelfPlayConfig(
            max_plies=state.ply + 1,
            repetition_target_transform=True,
        ),
    )

    assert default_game.samples[0].selected_move == repeating
    assert not default_game.samples[0].repetition_redirected
    assert legacy_game.samples[0].selected_move == alternative
    assert legacy_game.samples[0].repetition_redirected


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

    decision = transform_repetition_target(
        search,
        rules,
        state,
        repeating,
        temperature=0.0,
        value_tolerance=0.05,
        minimum_repeating_policy_mass=0.10,
        rng=random.Random(1),
    )

    assert decision.transformed
    assert decision.selected_move == alternative
    assert tuple(item.move for item in decision.policy_moves) == (alternative,)


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

    decision = transform_repetition_target(
        search,
        rules,
        state,
        repeating,
        temperature=0.0,
        value_tolerance=0.05,
        minimum_repeating_policy_mass=0.10,
        rng=random.Random(1),
    )

    assert not decision.transformed
    assert decision.defensive_repetition_preserved
    assert decision.selected_move == repeating
    assert decision.policy_moves == search.moves


def test_repetition_target_redirects_meaningful_mass_when_repeat_not_selected() -> None:
    rules = PythonChessRules()
    state = _threefold_choice_state(rules)
    repeating = ChessMove("f6g8")
    selected = ChessMove("h8g8")
    other = ChessMove("f6h5")
    search = SearchResult(
        (
            MoveStatistics(selected, 10, 0.5, 0.02),
            MoveStatistics(repeating, 5, 0.3, 0.0),
            MoveStatistics(other, 1, 0.2, -0.4),
        ),
        0.0,
        16,
    )

    decision = transform_repetition_target(
        search,
        rules,
        state,
        selected,
        temperature=0.0,
        value_tolerance=0.05,
        minimum_repeating_policy_mass=0.10,
        rng=random.Random(1),
    )

    assert decision.transformed
    assert decision.repeating_policy_mass == pytest.approx(5 / 16)
    assert decision.selected_move == selected
    assert tuple(item.move for item in decision.policy_moves) == (selected,)
