from pathlib import Path

import pytest

from harbichess.checkpoints.manifest import CheckpointManifest


def manifest() -> CheckpointManifest:
    return CheckpointManifest(
        format_version=1,
        model_name="HarbiChess",
        training_step=100,
        source_commit="a" * 40,
        state_schema_version=1,
        action_schema_version=1,
        backend="mlx-metal",
        metrics={"policy_loss": 1.25},
    )


def test_manifest_round_trip() -> None:
    expected = manifest()
    assert CheckpointManifest.from_json(expected.to_json()) == expected


def test_manifest_writes_utf8_json(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    manifest().write(destination)
    assert CheckpointManifest.from_json(destination.read_text(encoding="utf-8")) == manifest()


def test_manifest_requires_full_source_commit() -> None:
    with pytest.raises(ValueError, match="full 40-character Git SHA"):
        CheckpointManifest(
            format_version=1,
            model_name="HarbiChess",
            training_step=0,
            source_commit="abc123",
            state_schema_version=1,
            action_schema_version=1,
            backend="mlx-metal",
            metrics={},
        )
