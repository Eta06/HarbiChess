"""Versioned metadata linking evaluated checkpoints to reproducible source."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Metadata stored beside every checkpoint promoted to a release."""

    format_version: int
    model_name: str
    training_step: int
    source_commit: str
    state_schema_version: int
    action_schema_version: int
    backend: str
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        if self.format_version <= 0:
            raise ValueError("format_version must be positive")
        if self.training_step < 0:
            raise ValueError("training_step cannot be negative")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit.lower()
        ):
            raise ValueError("source_commit must be a full 40-character Git SHA")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> CheckpointManifest:
        data: dict[str, Any] = json.loads(payload)
        return cls(**data)

    def write(self, destination: Path) -> None:
        destination.write_text(self.to_json(), encoding="utf-8")

