import json
from pathlib import Path

import pytest

from harbichess.dashboard.state import (
    CheckpointStatus,
    PilotStatus,
    SnapshotStore,
)
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
        ),
        source_commit="a" * 40,
    )

    snapshot = SnapshotStore(telemetry_path).read()
    payload = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    assert result.passed
    assert snapshot.pilot_status is PilotStatus.PASSED
    assert snapshot.checkpoint_status is CheckpointStatus.VERIFIED
    assert snapshot.checkpoint_verified
    assert snapshot.diversity.games == 4
    assert snapshot.diversity.terminations[0].termination == "max_plies"
    assert snapshot.replay_shards == 2
    assert snapshot.training_step == result.training_steps
    assert snapshot.history[-1].training_step == 2
    assert payload["checkpoint"]["verified"]
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
    assert OcakRunConfig(run_id="qualified-teacher", teacher_oracle_depth=1)
    with pytest.raises(ValueError, match="safe path"):
        OcakRunConfig(run_id="../escape")
    with pytest.raises(ValueError, match="teacher_oracle_depth"):
        OcakRunConfig(run_id="invalid-teacher", teacher_oracle_depth=0)


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
        ),
        source_commit="b" * 40,
    )

    assert not result.passed
    assert "decisive terminal" in " ".join(result.reasons)
    assert "max-ply" in " ".join(result.reasons)
