"""Checksummed update-boundary checkpoints for continuous policy iteration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx


class ContinuousCheckpointIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ContinuousResumeState:
    update: int
    learner_step: int
    next_update_seed: int
    source_commit: str
    config_sha256: str
    model_file: str
    optimizer_file: str
    rolling_replay_files: tuple[str, ...]
    rolling_target_files: tuple[str, ...]
    artifact_sha256: dict[str, str]

    def __post_init__(self) -> None:
        if min(self.update, self.learner_step, self.next_update_seed) < 0:
            raise ValueError("continuous resume counters cannot be negative")
        if not self.source_commit or len(self.config_sha256) != 64:
            raise ValueError("continuous resume provenance is incomplete")


def save_continuous_resume(
    checkpoint_dir: Path,
    *,
    update: int,
    learner_step: int,
    next_update_seed: int,
    source_commit: str,
    config_sha256: str,
    optimizer_state: tuple[tuple[str, mx.array], ...],
    rolling_replay_files: tuple[Path, ...],
    rolling_target_files: tuple[Path, ...],
) -> ContinuousResumeState:
    """Atomically add optimizer/provenance state to an immutable model directory."""

    model_path = checkpoint_dir / "model.safetensors"
    optimizer_path = checkpoint_dir / "optimizer.safetensors"
    resume_path = checkpoint_dir / "resume.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"continuous checkpoint model is missing: {model_path}")
    if optimizer_path.exists() or resume_path.exists():
        raise FileExistsError(f"continuous resume state already exists: {checkpoint_dir}")
    referenced = (*rolling_replay_files, *rolling_target_files)
    if any(not path.is_file() for path in referenced):
        raise FileNotFoundError("continuous resume references a missing rolling artifact")

    temporary_optimizer = checkpoint_dir / ".optimizer.tmp.safetensors"
    temporary_resume = checkpoint_dir / ".resume.tmp.json"
    try:
        mx.save_safetensors(str(temporary_optimizer), dict(optimizer_state))
        os.replace(temporary_optimizer, optimizer_path)
        artifacts = (model_path, optimizer_path, *referenced)
        checksums = {str(path): _sha256(path) for path in artifacts}
        state = ContinuousResumeState(
            update=update,
            learner_step=learner_step,
            next_update_seed=next_update_seed,
            source_commit=source_commit,
            config_sha256=config_sha256,
            model_file=str(model_path),
            optimizer_file=str(optimizer_path),
            rolling_replay_files=tuple(str(path) for path in rolling_replay_files),
            rolling_target_files=tuple(str(path) for path in rolling_target_files),
            artifact_sha256=checksums,
        )
        temporary_resume.write_text(
            json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_resume, resume_path)
        return state
    finally:
        temporary_optimizer.unlink(missing_ok=True)
        temporary_resume.unlink(missing_ok=True)


def load_continuous_resume(
    checkpoint_dir: Path,
) -> tuple[ContinuousResumeState, tuple[tuple[str, mx.array], ...]]:
    payload = json.loads((checkpoint_dir / "resume.json").read_text(encoding="utf-8"))
    payload["rolling_replay_files"] = tuple(payload["rolling_replay_files"])
    payload["rolling_target_files"] = tuple(payload["rolling_target_files"])
    state = ContinuousResumeState(**payload)
    expected = {
        state.model_file,
        state.optimizer_file,
        *state.rolling_replay_files,
        *state.rolling_target_files,
    }
    if set(state.artifact_sha256) != expected:
        raise ContinuousCheckpointIntegrityError(
            "continuous resume manifest does not cover every artifact"
        )
    for filename, expected_sha256 in state.artifact_sha256.items():
        path = Path(filename)
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise ContinuousCheckpointIntegrityError(
                f"continuous checkpoint artifact failed verification: {filename}"
            )
    optimizer = tuple(mx.load(state.optimizer_file).items())
    mx.eval([value for _, value in optimizer])
    return state, optimizer


__all__ = [
    "ContinuousCheckpointIntegrityError",
    "ContinuousResumeState",
    "load_continuous_resume",
    "save_continuous_resume",
]
