from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from harbichess.training.continuous_checkpoint import (  # noqa: E402
    ContinuousCheckpointIntegrityError,
    load_continuous_resume,
    save_continuous_resume,
)


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "checkpoints" / "update-001"
    checkpoint.mkdir(parents=True)
    model = checkpoint / "model.safetensors"
    replay = tmp_path / "replay" / "update-001.jsonl.gz"
    target = tmp_path / "targets" / "update-001.json"
    replay.parent.mkdir()
    target.parent.mkdir()
    mx.save_safetensors(str(model), {"weight": mx.array([1.0, 2.0])})
    replay.write_bytes(b"replay")
    target.write_text("{}\n", encoding="utf-8")
    return checkpoint, replay, target


def test_continuous_resume_round_trips_and_covers_rolling_artifacts(tmp_path: Path) -> None:
    checkpoint, replay, target = _artifacts(tmp_path)

    saved = save_continuous_resume(
        checkpoint,
        update=1,
        learner_step=7,
        next_update_seed=42,
        source_commit="abc123",
        config_sha256="f" * 64,
        optimizer_state=(("step", mx.array(7)),),
        rolling_replay_files=(replay,),
        rolling_target_files=(target,),
    )
    loaded, optimizer = load_continuous_resume(checkpoint)

    assert loaded == saved
    assert dict(optimizer)["step"].item() == 7
    assert set(loaded.artifact_sha256) == {
        str(checkpoint / "model.safetensors"),
        str(checkpoint / "optimizer.safetensors"),
        str(replay),
        str(target),
    }


def test_continuous_resume_rejects_tampered_replay(tmp_path: Path) -> None:
    checkpoint, replay, target = _artifacts(tmp_path)
    save_continuous_resume(
        checkpoint,
        update=1,
        learner_step=7,
        next_update_seed=42,
        source_commit="abc123",
        config_sha256="f" * 64,
        optimizer_state=(),
        rolling_replay_files=(replay,),
        rolling_target_files=(target,),
    )
    replay.write_bytes(b"tampered")

    with pytest.raises(ContinuousCheckpointIntegrityError, match="failed verification"):
        load_continuous_resume(checkpoint)
