"""Backend-neutral policy/value inference contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EncodedPosition:
    """A flattened, versioned position tensor independent of ML frameworks."""

    values: tuple[float, ...]
    shape: tuple[int, ...]
    schema_version: int

    def __post_init__(self) -> None:
        expected = 1
        for dimension in self.shape:
            if dimension <= 0:
                raise ValueError("encoded position dimensions must be positive")
            expected *= dimension
        if expected != len(self.values):
            raise ValueError(f"shape requires {expected} values, got {len(self.values)}")


@dataclass(frozen=True, slots=True)
class PolicyValueOutput:
    """Raw policy logits and WDL logits for one encoded position."""

    policy_logits: tuple[float, ...]
    wdl_logits: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Execution features exposed for measurement and scheduling."""

    name: str
    device: str
    supports_training: bool
    supports_compilation: bool


@runtime_checkable
class PolicyValueBackend(Protocol):
    """The only interface search and self-play need from a neural backend."""

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def evaluate(self, positions: Sequence[EncodedPosition]) -> list[PolicyValueOutput]: ...
