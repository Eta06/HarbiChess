import json
import random
from pathlib import Path

import pytest

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, Side
from harbichess.dashboard.state import RunMode, SnapshotStore
from harbichess.evaluation.arena import ArenaConfig, play_arena_game, run_devir_arena
from harbichess.search.mcts import MoveStatistics, SearchResult

mx = pytest.importorskip("mlx.core")


class ScriptedArenaSearch:
    moves = (ChessMove("f2f3"), ChessMove("e7e5"), ChessMove("g2g4"), ChessMove("d8h4"))

    def search(self, state, *, rng: random.Random, add_root_noise: bool):
        del rng, add_root_noise
        move = self.moves[state.ply]
        return SearchResult((MoveStatistics(move, 1, 1.0, 0.0),), 0.0, 1)


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
    assert not snapshot.promotion_ready
    assert snapshot.arena_max_ply_draws == 2
    assert snapshot.arena_threefold_repetitions == 0
    assert "champion unchanged" in snapshot.mode_detail
    assert Path(result.result_path).is_file()
