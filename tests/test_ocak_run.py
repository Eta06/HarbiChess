import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.dashboard.state import (
    CheckpointStatus,
    PilotStatus,
    SnapshotStore,
    empty_snapshot,
)
from harbichess.replay.coverage import ReplayCoverageThresholds
from harbichess.replay.split import ReplaySplit, split_for_game
from harbichess.training.ocak_run import OcakRunConfig, run_ocak_sanity

pytest.importorskip("mlx.core")


def _run_id_with_both_splits(game_count: int) -> str:
    for suffix in range(100):
        run_id = f"test-ocak-{suffix}"
        splits = {
            split_for_game(
                f"{run_id}-{index:012d}",
                validation_fraction=0.5,
            )
            for index in range(game_count)
        }
        if splits == {ReplaySplit.TRAIN, ReplaySplit.VALIDATION}:
            return run_id
    raise AssertionError("could not find a deterministic test split")


def test_ocak_run_connects_self_play_training_checkpoint_and_telemetry(
    tmp_path: Path,
) -> None:
    run_id = _run_id_with_both_splits(4)
    telemetry_path = tmp_path / "dashboard.json"
    SnapshotStore(telemetry_path).write_atomic(
        replace(
            empty_snapshot(),
            teacher_qualification_status="passed",
            teacher_best_variant="oracle-512",
        )
    )
    result = run_ocak_sanity(
        OcakRunConfig(
            run_id=run_id,
            artifact_root=tmp_path / "runs",
            telemetry_path=telemetry_path,
            run_seed=19,
            games=4,
            workers=4,
            simulations=1,
            max_plies=1,
            exploration_plies=1,
            validation_fraction=0.5,
            training_steps=2,
            batch_size=2,
            learning_rate=0.002,
            minimum_train_improvement=0.0,
            maximum_validation_ratio=100.0,
            minimum_decisive_games=0,
            maximum_max_ply_draw_ratio=1.0,
            telemetry_interval_steps=1,
            trunk_channels=8,
            residual_blocks=1,
            policy_channels=2,
            value_channels=1,
            value_hidden=8,
            tactical_gate_budgets=(),
        ),
        source_commit="a" * 40,
    )

    snapshot = SnapshotStore(telemetry_path).read()
    payload = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    assert not result.passed
    assert snapshot.pilot_status is PilotStatus.FAILED
    assert snapshot.checkpoint_status is CheckpointStatus.VERIFIED
    assert snapshot.checkpoint_verified
    assert snapshot.teacher_qualification_status == "passed"
    assert snapshot.teacher_best_variant == "oracle-512"
    assert snapshot.diversity.games == 4
    assert snapshot.diversity.terminations[0].termination == "max_plies"
    assert snapshot.replay_shards == 2
    assert snapshot.training_step == result.training_steps == 0
    assert snapshot.history[-1].training_step == 2
    assert payload["checkpoint"]["verified"]
    assert payload["loss"]["validation_value_samples"] == 0
    assert "no known value targets" in " ".join(result.reasons)
    assert Path(payload["baseline"]["path"]).is_file()
    assert len(payload["baseline"]["model_sha256"]) == 64
    assert Path(result.checkpoint_path, "resume.json").is_file()


def test_ocak_run_configuration_rejects_unsafe_run_id() -> None:
    clean = OcakRunConfig(run_id="clean-default")
    assert not clean.repetition_target_transform
    assert clean.teacher_oracle_depth is None
    assert clean.learning_rate == pytest.approx(0.0002)
    assert OcakRunConfig(
        run_id="legacy-reproduction",
        repetition_target_transform=True,
    ).repetition_target_transform
    assert OcakRunConfig(
        run_id="qualified-teacher",
        teacher_oracle_depth=1,
        teacher_oracle_workers=4,
    ).teacher_oracle_workers == 4
    with pytest.raises(ValueError, match="safe path"):
        OcakRunConfig(run_id="../escape")
    with pytest.raises(ValueError, match="teacher_oracle_depth"):
        OcakRunConfig(run_id="invalid-teacher", teacher_oracle_depth=0)
    with pytest.raises(ValueError, match="workers"):
        OcakRunConfig(run_id="orphan-workers", teacher_oracle_workers=1)
    with pytest.raises(ValueError, match="qualification"):
        OcakRunConfig(run_id="ungated-generation", generation_only=True)


def test_generation_only_run_writes_replay_without_starting_learner(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    HarbiChessNetwork(
        NetworkConfig(
            trunk_channels=8,
            residual_blocks=1,
            policy_channels=2,
            value_channels=1,
            value_hidden=8,
        )
    ).save_weights(str(model))
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    qualification = tmp_path / "teacher.json"
    qualification.write_text(
        json.dumps(
            {
                "baseline": {"model_sha256": model_hash},
                "config": {"oracle_depth": 1},
                "gate": {"qualified_oracle_budgets": [1]},
            }
        ),
        encoding="utf-8",
    )
    permissive = ReplayCoverageThresholds(
        minimum_samples=1,
        minimum_unique_position_ratio=0.0,
        minimum_opening_ratio=0.0,
        minimum_middlegame_ratio=0.0,
        minimum_endgame_ratio=0.0,
        minimum_tactical_ratio=0.0,
        minimum_quiet_ratio=0.0,
        minimum_value_bucket_ratio=0.0,
        minimum_outcome_bucket_ratio=0.0,
        minimum_material_signatures=1,
        minimum_position_signatures=1,
        minimum_teacher_telemetry_ratio=1.0,
        minimum_comparable_teacher_deltas=0,
        minimum_positive_teacher_delta_ratio=0.0,
        minimum_mean_teacher_delta=-1.0,
    )
    result = run_ocak_sanity(
        OcakRunConfig(
            run_id=_run_id_with_both_splits(4),
            artifact_root=tmp_path / "runs",
            telemetry_path=tmp_path / "dashboard.json",
            games=4,
            workers=2,
            simulations=1,
            max_plies=1,
            validation_fraction=0.5,
            initial_model=model,
            trunk_channels=8,
            residual_blocks=1,
            policy_channels=2,
            value_channels=1,
            value_hidden=8,
            teacher_oracle_depth=1,
            generation_only=True,
            teacher_qualification_result=qualification,
            replay_coverage_thresholds=permissive,
        ),
        source_commit="c" * 40,
    )

    payload = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    snapshot = SnapshotStore(tmp_path / "dashboard.json").read()
    assert result.passed
    assert result.training_steps == 0
    assert result.checkpoint_path == ""
    assert payload["mode"] == "generation_only"
    assert payload["checkpoint"] is None
    assert payload["coverage"]["passed"]
    assert snapshot.pilot_status is PilotStatus.REPLAY
    assert snapshot.candidate_checkpoint == "None"


def test_ocak_run_rejects_draw_only_truncated_self_play(tmp_path: Path) -> None:
    result = run_ocak_sanity(
        OcakRunConfig(
            run_id=_run_id_with_both_splits(4),
            artifact_root=tmp_path / "runs",
            telemetry_path=tmp_path / "dashboard.json",
            games=4,
            workers=4,
            simulations=1,
            max_plies=1,
            validation_fraction=0.5,
            training_steps=1,
            batch_size=1,
            minimum_train_improvement=0.0,
            maximum_validation_ratio=100.0,
            trunk_channels=8,
            residual_blocks=1,
            policy_channels=2,
            value_channels=1,
            value_hidden=8,
            tactical_gate_budgets=(),
        ),
        source_commit="b" * 40,
    )

    assert not result.passed
    assert "decisive terminal" in " ".join(result.reasons)
    assert "max-ply" in " ".join(result.reasons)
