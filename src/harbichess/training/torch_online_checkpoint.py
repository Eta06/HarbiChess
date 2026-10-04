"""Atomic native checkpoints for an experimental CPU online learner.

Current, frozen base and EMA networks, AdamW, every used RNG and a complete
actor cursor publish together. This format is separate from legacy checkpoints
and weights-only imports. The actor must validate its versioned run_state.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256
from harbichess.chess.actions import ACTION_SCHEMA_VERSION
from harbichess.chess.encoding import ENCODER_SCHEMA_VERSION
from harbichess.training.online_objective import ONLINE_OBJECTIVE_SCHEMA
from harbichess.training.online_targets import ONLINE_TARGET_SCHEMA

ONLINE_CHECKPOINT_SCHEMA = "torch-online-native-v1"
_FILES = ("model.safetensors", "base.safetensors", "ema.safetensors", "training.pt", "actor.json")
_SCHEMAS = {
    "encoder": ENCODER_SCHEMA_VERSION,
    "actions": ACTION_SCHEMA_VERSION,
    "target": ONLINE_TARGET_SCHEMA,
    "objective": ONLINE_OBJECTIVE_SCHEMA,
}


class OnlineCheckpointIntegrityError(ValueError):
    pass


def _runtime() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }


def _validate_models(online, base, ema):
    if online.specification != base.specification or online.specification != ema.specification:
        raise ValueError("current/base/EMA architectures must match")
    if online is base or online is ema or base is ema:
        raise ValueError("current/base/EMA networks must be separate owned instances")
    for model in (online, base, ema):
        for parameter in model.parameters():
            if (
                parameter.device.type != "cpu"
                or parameter.dtype != torch.float32
                or not torch.isfinite(parameter).all()
            ):
                raise ValueError("native online checkpoints require finite CPU float32 networks")
    if any(parameter.requires_grad for model in (base, ema) for parameter in model.parameters()):
        raise ValueError("base and EMA networks must be frozen")
    storages = [
        {p.untyped_storage().data_ptr() for p in model.parameters()}
        for model in (online, base, ema)
    ]
    if any(storages[i] & storages[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("current/base/EMA parameter storage must not alias")


def _validate_optimizer(optimizer, online, update):
    trainable = [p for p in online.parameters() if p.requires_grad]
    if (
        type(optimizer) is not torch.optim.AdamW
        or len(optimizer.param_groups) != 1
        or [id(p) for p in optimizer.param_groups[0]["params"]] != [id(p) for p in trainable]
        or not trainable
    ):
        raise ValueError("online AdamW requires exactly the ordered trainable parameter group")
    group = optimizer.param_groups[0]
    if (
        not math.isfinite(group["lr"])
        or group["lr"] <= 0
        or not math.isfinite(group["weight_decay"])
        or group["weight_decay"] < 0
        or not math.isfinite(group["eps"])
        or group["eps"] <= 0
        or any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in group["betas"])
    ):
        raise ValueError("invalid online AdamW hyperparameters")
    allowed = {"step", "exp_avg", "exp_avg_sq"}
    if group["amsgrad"]:
        allowed.add("max_exp_avg_sq")
    for parameter, state in optimizer.state.items():
        if not any(parameter is p for p in trainable) or set(state) != allowed:
            raise ValueError("invalid online AdamW state names or parameter ownership")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError("online AdamW state must contain finite tensors")
            if name == "step":
                if value.numel() != 1:
                    raise ValueError("online AdamW step counter must be scalar")
                step = float(value)
                if not step.is_integer() or not 0 <= step <= update:
                    raise ValueError("online AdamW step counter differs from update cursor")
            elif value.shape != parameter.shape or value.device.type != "cpu":
                raise ValueError("online AdamW moment shape/device differs from parameter")


def save_online_checkpoint(
    directory: Path,
    *,
    online: TorchChessNetwork,
    base: TorchChessNetwork,
    ema: TorchChessNetwork,
    optimizer: torch.optim.AdamW,
    actor_rng: random.Random,
    run_state: dict,
    run_config: dict,
    input_paths: dict[str, Path],
    source_commit: str,
    update: int,
) -> dict:
    if directory.exists():
        raise FileExistsError(directory)
    if (
        len(source_commit) != 40
        or any(char not in "0123456789abcdef" for char in source_commit)
        or type(update) is not int
        or update < 0
        or run_state.get("update") != update
    ):
        raise ValueError("invalid source commit or online update cursor")
    _validate_models(online, base, ema)
    _validate_optimizer(optimizer, online, update)
    trainable = [(name, p) for name, p in online.named_parameters() if p.requires_grad]
    actor_json = json.dumps(run_state, indent=2, sort_keys=True, allow_nan=False) + "\n"
    config = json.loads(json.dumps(run_config, allow_nan=False))
    inputs = {
        name: {
            "path": os.path.relpath(path.resolve(), directory.resolve()),
            "sha256": sha256(path),
        }
        for name, path in input_paths.items()
    }
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=directory.parent, prefix=f".{directory.name}."))
    try:
        for name, model in zip(_FILES[:3], (online, base, ema), strict=True):
            save_weights(
                temporary / name,
                model,
                provenance={"source_commit": source_commit, "online_update": update},
            )
        numpy_state = np.random.get_state()
        torch.save(
            {
                "update": update,
                "optimizer": optimizer.state_dict(),
                "actor_rng": actor_rng.getstate(),
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": {
                    "kind": numpy_state[0],
                    "keys": torch.tensor(numpy_state[1].astype(np.int64)),
                    "position": numpy_state[2],
                    "has_gauss": numpy_state[3],
                    "cached_gauss": numpy_state[4],
                },
            },
            temporary / "training.pt",
        )
        (temporary / "actor.json").write_text(actor_json, encoding="utf-8")
        manifest = {
            "schema": ONLINE_CHECKPOINT_SCHEMA,
            "backend": "torch",
            "transfer": "full-online-training",
            "schemas": _SCHEMAS,
            "source_commit": source_commit,
            "update": update,
            "run_config": config,
            "runtime": _runtime(),
            "trainable": [name for name, _ in trainable],
            "model_modes": [model.training for model in (online, base, ema)],
            "inputs": inputs,
            "artifacts": {name: sha256(temporary / name) for name in _FILES},
        }
        (temporary / "checkpoint.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for path in temporary.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(temporary, directory)
        descriptor = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@dataclass
class OnlineCheckpoint:
    online: TorchChessNetwork
    base: TorchChessNetwork
    ema: TorchChessNetwork
    optimizer: torch.optim.AdamW
    actor_rng: random.Random
    run_state: dict
    manifest: dict


def load_online_checkpoint(
    directory: Path,
    *,
    expected_run_config: dict,
    expected_input_paths: dict[str, Path],
    expected_source_commit: str,
) -> OnlineCheckpoint:
    manifest = json.loads((directory / "checkpoint.json").read_text())
    if (
        manifest["schema"] != ONLINE_CHECKPOINT_SCHEMA
        or manifest["backend"] != "torch"
        or manifest["transfer"] != "full-online-training"
        or manifest["schemas"] != _SCHEMAS
        or manifest["run_config"] != expected_run_config
        or manifest["source_commit"] != expected_source_commit
        or manifest["runtime"] != _runtime()
    ):
        raise OnlineCheckpointIntegrityError("online schema/config/source/runtime mismatch")
    if set(manifest["artifacts"]) != set(_FILES):
        raise OnlineCheckpointIntegrityError("incomplete online checkpoint artifacts")
    for name, digest in manifest["artifacts"].items():
        if not (directory / name).is_file() or sha256(directory / name) != digest:
            raise OnlineCheckpointIntegrityError(f"online artifact checksum mismatch: {name}")
    if set(manifest["inputs"]) != set(expected_input_paths):
        raise OnlineCheckpointIntegrityError("online immutable input names differ")
    for name, item in manifest["inputs"].items():
        path = (directory / item["path"]).resolve()
        if (
            path != expected_input_paths[name].resolve()
            or not path.is_file()
            or sha256(path) != item["sha256"]
        ):
            raise OnlineCheckpointIntegrityError(f"online immutable input mismatch: {name}")
    run_state = json.loads((directory / "actor.json").read_text())
    state = torch.load(directory / "training.pt", map_location="cpu", weights_only=True)
    if state["update"] != manifest["update"] or run_state.get("update") != manifest["update"]:
        raise OnlineCheckpointIntegrityError("online update cursor mismatch")
    online, base, ema = (load_weights(directory / name) for name in _FILES[:3])
    trainable = set(manifest["trainable"])
    if not trainable or not trainable <= dict(online.named_parameters()).keys():
        raise OnlineCheckpointIntegrityError("online trainable parameter names differ")
    for name, parameter in online.named_parameters():
        parameter.requires_grad_(name in trainable)
    for model in (base, ema):
        model.requires_grad_(False)
    _validate_models(online, base, ema)
    optimizer = torch.optim.AdamW([p for p in online.parameters() if p.requires_grad], lr=1e-4)
    if len(state["optimizer"]["param_groups"]) != 1:
        raise OnlineCheckpointIntegrityError("online AdamW group count differs")
    optimizer.load_state_dict(state["optimizer"])
    try:
        _validate_optimizer(optimizer, online, manifest["update"])
    except ValueError as error:
        raise OnlineCheckpointIntegrityError(str(error)) from error
    actor_rng = random.Random(0)
    actor_rng.setstate(state["actor_rng"])
    # Validate every RNG and mode before changing process-global RNG states.
    random.Random(0).setstate(state["python_rng"])
    numpy_rng = state["numpy_rng"]
    numpy_state = (
        numpy_rng["kind"],
        numpy_rng["keys"].numpy().astype(np.uint32),
        numpy_rng["position"],
        numpy_rng["has_gauss"],
        numpy_rng["cached_gauss"],
    )
    np.random.RandomState(0).set_state(numpy_state)
    torch.Generator(device="cpu").set_state(state["torch_rng"])
    if len(manifest["model_modes"]) != 3 or any(
        type(mode) is not bool for mode in manifest["model_modes"]
    ):
        raise OnlineCheckpointIntegrityError("invalid current/base/EMA modes")
    for model, mode in zip((online, base, ema), manifest["model_modes"], strict=True):
        model.train(mode)
    random.setstate(state["python_rng"])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_rng"])
    return OnlineCheckpoint(online, base, ema, optimizer, actor_rng, run_state, manifest)
