"""Publish an evaluated checkpoint as a commit-linked GitHub Release."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path

from harbichess.checkpoints.manifest import CheckpointManifest

MAX_RELEASE_ASSET_BYTES = 2 * 1024**3 - 1


def validate_assets(checkpoint: Path, manifest_path: Path, source_commit: str) -> None:
    """Fail before publication if assets are missing, oversized, or mismatched."""

    for asset in (checkpoint, manifest_path):
        if not asset.is_file():
            raise ValueError(f"release asset does not exist: {asset}")
        if asset.stat().st_size > MAX_RELEASE_ASSET_BYTES:
            raise ValueError(f"release asset must be smaller than 2 GiB: {asset}")

    manifest = CheckpointManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.source_commit != source_commit:
        raise ValueError(
            "checkpoint manifest source_commit does not match the release target: "
            f"{manifest.source_commit} != {source_commit}"
        )


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def publish(tag: str, checkpoint: Path, manifest_path: Path, *, title: str) -> None:
    source_commit = current_commit()
    validate_assets(checkpoint, manifest_path, source_commit)
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            tag,
            str(checkpoint),
            str(manifest_path),
            "--target",
            source_commit,
            "--title",
            title,
            "--generate-notes",
        ],
        check=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="model release tag, for example model-v0.01")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--title", default="HarbiChess evaluated checkpoint")
    arguments = parser.parse_args(argv)
    publish(arguments.tag, arguments.checkpoint, arguments.manifest, title=arguments.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

