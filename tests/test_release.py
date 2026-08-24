from pathlib import Path

import pytest

from harbichess.checkpoints.manifest import CheckpointManifest
from harbichess.checkpoints.release import validate_assets


def write_manifest(path: Path, source_commit: str) -> None:
    CheckpointManifest(
        format_version=1,
        model_name="HarbiChess",
        training_step=100,
        source_commit=source_commit,
        state_schema_version=1,
        action_schema_version=1,
        backend="mlx-metal",
        metrics={},
    ).write(path)


def test_release_assets_must_match_source_commit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    manifest = tmp_path / "manifest.json"
    checkpoint.write_bytes(b"weights")
    write_manifest(manifest, "a" * 40)

    validate_assets(checkpoint, manifest, "a" * 40)
    with pytest.raises(ValueError, match="does not match"):
        validate_assets(checkpoint, manifest, "b" * 40)


def test_release_assets_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        validate_assets(tmp_path / "missing", tmp_path / "manifest.json", "a" * 40)
