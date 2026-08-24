"""Crash-safe metadata required to resume a HarbiChess training run."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Links model, optimizer, replay, RNG, counters, and elapsed time."""

    schema_version: int
    run_id: str
    checkpoint_id: str
    source_commit: str
    created_at: str
    training_step: int
    lifetime_games: int
    generation_games: int
    training_elapsed_seconds: float
    replay_samples: int
    replay_cursor: int
    model_file: str
    optimizer_file: str
    rng_file: str
    artifact_sha256: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit.lower()
        ):
            raise ValueError("source_commit must be a full 40-character Git SHA")
        counters = (
            self.training_step,
            self.lifetime_games,
            self.generation_games,
            self.training_elapsed_seconds,
            self.replay_samples,
            self.replay_cursor,
        )
        if any(value < 0 for value in counters):
            raise ValueError("resume counters cannot be negative")
        artifacts = (self.model_file, self.optimizer_file, self.rng_file)
        if not self.run_id or not self.checkpoint_id or not all(artifacts):
            raise ValueError("resume identifiers and artifact files cannot be empty")
        if any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
            for digest in self.artifact_sha256.values()
        ):
            raise ValueError("artifact checksums must be 64-character SHA-256 digests")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> ResumeState:
        data: dict[str, Any] = json.loads(payload)
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> ResumeState:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save_atomic(self, path: Path) -> None:
        """Replace the resume pointer only after the new state is durable."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(self.to_json())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
