"""Immutable versioned CPU/CUDA training checkpoints with verified replay and RNG.

The entire directory publishes in one rename. Moving the complete run directory
preserves relative replay references. Weights-only imports have no resume manifest.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from harbichess.backends.torch_network import load_weights, save_weights, sha256
from harbichess.replay.schema import SCHEMA_VERSIONS
from harbichess.training.batch import GameBalancedSampler
from harbichess.training.config import LearnerConfig
from harbichess.training.torch_learner import TorchLearner

CHECKPOINT_SCHEMA = 1


class TorchCheckpointIntegrityError(ValueError):
    pass


def runtime_signature(device: torch.device) -> dict:
    return {
        "torch": torch.__version__,
        "device": str(device),
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }


def save_checkpoint(
    directory: Path,
    *,
    learner: TorchLearner,
    sampler: GameBalancedSampler | None,
    replay_paths: tuple[Path, ...],
    run_state: dict,
    run_config: dict,
    source_commit: str,
) -> dict:
    if directory.exists():
        raise FileExistsError(directory)
    if len(source_commit) != 40:
        raise ValueError("source_commit must be a full Git SHA")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=directory.parent, prefix=f".{directory.name}."))
    try:
        save_weights(
            temporary / "model.safetensors",
            learner.network,
            provenance={"source_commit": source_commit, "learner_step": learner.step},
        )
        state = {
            "optimizer": learner.optimizer.state_dict(),
            "step": learner.step,
            "sampler_rng": sampler.rng_state if sampler else None,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if learner.device.type == "cuda" else [],
        }
        torch.save(state, temporary / "training.pt")
        replay = {
            os.path.relpath(p.resolve(), directory.resolve()): sha256(p) for p in replay_paths
        }
        manifest = {
            "schema": CHECKPOINT_SCHEMA,
            "backend": "torch",
            "transfer": "full-training",
            "schemas": SCHEMA_VERSIONS,
            "source_commit": source_commit,
            "learner_config": asdict(learner.config),
            "run_config": run_config,
            "run_state": run_state,
            "step": learner.step,
            "trainable": [k for k, v in learner.network.named_parameters() if v.requires_grad],
            "runtime": runtime_signature(learner.device),
            "replay": replay,
            "artifacts": {
                name: sha256(temporary / name) for name in ("model.safetensors", "training.pt")
            },
        }
        (temporary / "checkpoint.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        for path in temporary.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(temporary, directory)
        fd = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_checkpoint(
    directory: Path, *, expected_run_config: dict, device: str = "cpu"
) -> tuple[TorchLearner, dict, object]:
    manifest = json.loads((directory / "checkpoint.json").read_text())
    if (
        manifest["schema"] != CHECKPOINT_SCHEMA
        or manifest["backend"] != "torch"
        or manifest["transfer"] != "full-training"
        or manifest["schemas"] != SCHEMA_VERSIONS
    ):
        raise TorchCheckpointIntegrityError("unsupported training checkpoint/schema")
    if manifest["run_config"] != expected_run_config:
        raise TorchCheckpointIntegrityError("run configuration mismatch; resume refused")
    if manifest["runtime"] != runtime_signature(torch.device(device)):
        raise TorchCheckpointIntegrityError(
            "runtime mismatch; exact resume requires recorded runtime"
        )
    if set(manifest["artifacts"]) != {"model.safetensors", "training.pt"}:
        raise TorchCheckpointIntegrityError("incomplete training manifest")
    for filename, digest in {**manifest["artifacts"], **manifest["replay"]}.items():
        path = directory / filename
        if not path.is_file() or sha256(path) != digest:
            raise TorchCheckpointIntegrityError(f"artifact checksum mismatch: {filename}")
    network = load_weights(directory / "model.safetensors")
    selected = set(manifest["trainable"])
    if not selected <= dict(network.named_parameters()).keys():
        raise TorchCheckpointIntegrityError("unknown trainable parameters")
    for k, v in network.named_parameters():
        v.requires_grad_(k in selected)
    learner = TorchLearner(
        network, config=LearnerConfig(**manifest["learner_config"]), device=device
    )
    # Reject arbitrary pickle code; optimizer/RNG use plain tensors and containers.
    state = torch.load(directory / "training.pt", map_location="cpu", weights_only=True)
    if state["step"] != manifest["step"]:
        raise TorchCheckpointIntegrityError("learner step mismatch")
    learner.optimizer.load_state_dict(state["optimizer"])
    learner.step = state["step"]
    torch.set_rng_state(state["torch_rng"])
    if learner.device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return learner, manifest, state["sampler_rng"]
