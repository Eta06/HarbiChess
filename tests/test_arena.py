import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, GameOutcome, Side, TerminalResult
from harbichess.dashboard.state import RunMode, SnapshotStore
from harbichess.evaluation.arena import (
    ArenaConfig,
    ArenaGame,
    ContinuationRoot,
    _continuation_records,
    _select_checkpoint,
    play_arena_game,
    run_devir_arena,
)
from harbichess.search.mcts import MoveStatistics, SearchResult

mx = pytest.importorskip("mlx.core")


def test_arena_selects_requested_validation_checkpoint() -> None:
    selected = {"path": "best", "manifest": {"checkpoint_id": "candidate-step-20"}}
    earlier = {"path": "early", "manifest": {"checkpoint_id": "candidate-step-10"}}
    payload = {"checkpoint": selected, "validation_checkpoints": [earlier, selected]}

    assert _select_checkpoint(payload, None) is selected
    assert _select_checkpoint(payload, "candidate-step-10") is earlier
    with pytest.raises(ValueError, match="unknown validation checkpoint"):
        _select_checkpoint(payload, "candidate-step-99")


class ScriptedArenaSearch:
    moves = (ChessMove("f2f3"), ChessMove("e7e5"), ChessMove("g2g4"), ChessMove("d8h4"))

    def search(self, state, *, rng: random.Random, add_root_noise: bool):
        del rng, add_root_noise
        move = self.moves[state.ply]
        return SearchResult((MoveStatistics(move, 1, 1.0, 0.0),), 0.0, 1)


class RepeatingArenaSearch:
    moves = (
        ChessMove("g1f3"),
        ChessMove("g8f6"),
        ChessMove("f3g1"),
        ChessMove("f6g8"),
    )

    def search(self, state, *, rng: random.Random, add_root_noise: bool):
        del rng, add_root_noise
        move = self.moves[state.ply % len(self.moves)]
        alternative = next(item for item in PythonChessRules().legal_moves(state) if item != move)
        return SearchResult(
            (
                MoveStatistics(move, 3, 0.75, 0.0),
                MoveStatistics(alternative, 1, 0.25, -0.01),
            ),
            0.0,
            4,
        )


def test_arena_scores_result_from_candidate_color() -> None:
    rules = PythonChessRules()
    game = play_arena_game(
        ScriptedArenaSearch(),
        ScriptedArenaSearch(),
        rules,
        rules.initial_state(),
        game_id="pair-0-black",
        pair_index=0,
        candidate_side=Side.BLACK,
        opening_moves=(),
        max_plies=20,
    )

    assert game.candidate_score == 1.0
    assert game.final_state.ply == 4
    assert game.candidate_side is Side.BLACK


def test_arena_marks_threefold_when_non_repeating_alternatives_exist() -> None:
    rules = PythonChessRules()
    game = play_arena_game(
        RepeatingArenaSearch(),
        RepeatingArenaSearch(),
        rules,
        rules.initial_state(),
        game_id="repetition",
        pair_index=0,
        candidate_side=Side.WHITE,
        opening_moves=(),
        max_plies=20,
    )

    assert game.outcome.termination == "threefold_repetition"
    assert game.avoidable_threefold
    assert len(game.continuation_roots) == 1
    assert game.continuation_roots[0].repeating_policy_mass == pytest.approx(0.75)


def test_arena_continuation_roots_become_legal_replay_targets() -> None:
    rules = PythonChessRules()
    state = rules.initial_state()
    root = ContinuationRoot(
        state=state,
        side_to_move=Side.WHITE,
        policy=((ChessMove("e2e4"), 0.75), (ChessMove("d2d4"), 0.25)),
        selected_move=ChessMove("e2e4"),
        root_value=0.1,
        repeating_policy_mass=0.4,
        source_model="candidate",
    )
    game = ArenaGame(
        game_id="arena-game",
        pair_index=0,
        candidate_side=Side.WHITE,
        opening_moves=(),
        final_state=rules.apply(state, ChessMove("e2e4")),
        outcome=GameOutcome(TerminalResult.DRAW, "threefold_repetition"),
        candidate_score=0.5,
        last_search=None,
        avoidable_threefold=True,
        continuation_roots=(root,),
    )

    records = _continuation_records(
        (game,),
        arena_id="arena",
        seed=7,
        rules=rules,
    )

    assert len(records) == 1
    assert records[0].repetition_redirected
    assert records[0].outcome_value == 0

    truncated = replace(
        game,
        outcome=GameOutcome(TerminalResult.DRAW, "max_plies"),
    )
    truncated_records = _continuation_records(
        (truncated,),
        arena_id="truncated-arena",
        seed=11,
        rules=rules,
    )
    assert truncated_records[0].outcome_value is None
    records[0].validate_rules(rules)


def test_arena_configuration_rejects_unsafe_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="safe path"):
        ArenaConfig(arena_id="../escape", ocak_result=tmp_path / "result.json")
    with pytest.raises(ValueError, match="positive"):
        ArenaConfig(arena_id="test", ocak_result=tmp_path / "result.json", opening_pairs=0)


def test_devir_runner_keeps_champion_when_micro_arena_is_inconclusive(
    tmp_path: Path,
) -> None:
    network_config = NetworkConfig(
        trunk_channels=8,
        residual_blocks=1,
        policy_channels=2,
        value_channels=1,
        value_hidden=8,
    )
    mx.random.seed(31)
    checkpoint = tmp_path / "checkpoints" / "candidate"
    checkpoint.mkdir(parents=True)
    HarbiChessNetwork(network_config).save_weights(str(checkpoint / "model.safetensors"))
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "passed": True,
                "source_commit": "a" * 40,
                "config": {
                    "run_seed": 31,
                    "trunk_channels": 8,
                    "residual_blocks": 1,
                    "policy_channels": 2,
                    "value_channels": 1,
                    "value_hidden": 8,
                },
                "checkpoint": {
                    "path": str(checkpoint),
                    "manifest": {
                        "checkpoint_id": "candidate",
                        "model_file": "model.safetensors",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    telemetry = tmp_path / "dashboard.json"

    result = run_devir_arena(
        ArenaConfig(
            arena_id="micro-arena",
            ocak_result=result_path,
            telemetry_path=telemetry,
            opening_pairs=1,
            opening_plies=0,
            simulations=1,
            max_plies=1,
            workers=2,
        )
    )

    snapshot = SnapshotStore(telemetry).read()
    assert result.games == 2
    assert result.draws == 2
    assert not result.promotion_ready
    assert snapshot.mode is RunMode.IDLE
    assert snapshot.candidate_checkpoint == "candidate"
    assert not snapshot.promotion_ready
    assert snapshot.arena_max_ply_draws == 2
    assert snapshot.arena_threefold_repetitions == 0
    assert snapshot.arena_avoidable_threefold_repetitions == 0
    assert "champion unchanged" in snapshot.mode_detail
    assert Path(result.result_path).is_file()
