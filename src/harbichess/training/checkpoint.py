"""Atomic, checksummed MLX learner checkpoints with exact sampler resume."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from harbichess.training.batch import GameBalancedSampler
from harbichess.training.learner import MLXLearner
from harbichess.training.resume import ResumeState


class CheckpointIntegrityError(RuntimeError):
    pass


def _artifact_path(directory: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != filename:
        raise ValueError("checkpoint artifact names must be plain filenames")
    return directory / filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return {"tuple": [_jsonable(item) for item in value]}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported RNG state value: {type(value).__name__}")


def _from_jsonable(value: object) -> object:
    if isinstance(value, dict):
        if set(value) != {"tuple"} or not isinstance(value["tuple"], list):
            raise ValueError("invalid tuple encoding in RNG state")
        return tuple(_from_jsonable(item) for item in value["tuple"])
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    return value


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def save_training_checkpoint(
    directory: Path,
    *,
    state: ResumeState,
    learner: MLXLearner,
    sampler: GameBalancedSampler,
) -> ResumeState:
    """Publish a complete immutable checkpoint directory in one rename."""

    if directory.exists():
        raise FileExistsError(f"checkpoint already exists: {directory}")
    if learner.step != state.training_step:
        raise ValueError("resume training_step must match the learner step")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        model_path = _artifact_path(temporary, state.model_file)
        optimizer_path = _artifact_path(temporary, state.optimizer_file)
        rng_path = _artifact_path(temporary, state.rng_file)
        learner.network.save_weights(str(model_path))
        mx.save_safetensors(str(optimizer_path), dict(tree_flatten(learner.optimizer.state)))
        rng_path.write_text(
            json.dumps(_jsonable(sampler.rng_state), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        for path in (model_path, optimizer_path, rng_path):
            _sync_file(path)
        checksums = {
            state.model_file: _sha256(model_path),
            state.optimizer_file: _sha256(optimizer_path),
            state.rng_file: _sha256(rng_path),
        }
        saved_state = replace(state, artifact_sha256=checksums)
        saved_state.save_atomic(temporary / "resume.json")
        os.rename(temporary, directory)
        parent_fd = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return saved_state
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_training_checkpoint(
    directory: Path,
    *,
    learner: MLXLearner,
    sampler: GameBalancedSampler,
) -> ResumeState:
    """Validate and restore model, optimizer, learner step, and sampler RNG."""

    state = ResumeState.load(directory / "resume.json")
    expected = {state.model_file, state.optimizer_file, state.rng_file}
    if set(state.artifact_sha256) != expected:
        raise CheckpointIntegrityError("checkpoint manifest does not cover every artifact")
    paths = {name: _artifact_path(directory, name) for name in expected}
    for name, path in paths.items():
        if not path.is_file() or _sha256(path) != state.artifact_sha256[name]:
            raise CheckpointIntegrityError(f"checkpoint artifact failed verification: {name}")

    learner.network.load_weights(str(paths[state.model_file]))
    optimizer_items: dict[str, Any] = dict(mx.load(str(paths[state.optimizer_file])))
    learner.optimizer.state = tree_unflatten(list(optimizer_items.items()))
    rng_payload = json.loads(paths[state.rng_file].read_text(encoding="utf-8"))
    sampler.set_rng_state(_from_jsonable(rng_payload))
    learner.step = state.training_step
    mx.eval(learner.network.parameters(), learner.optimizer.state)
    return state
