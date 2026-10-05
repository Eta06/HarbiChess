"""Independent ownsearch native format; saves only closed, empty replay boundaries."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from harbichess.backends.torch_network import save_weights, sha256
from harbichess.selfplay.online_actor import OnlineActors
from harbichess.selfplay.online_epoch import deserialize_policy_epoch
from harbichess.training.fullgame_own_targets import build_fullgame_targets
from harbichess.training.torch_online_checkpoint import _runtime
from harbichess.training.torch_online_learner import read_online_train_book

CPU_SCHEMA = "torch-ownsearch-native-cpu-v1"
CUDA_SCHEMA = "torch-ownsearch-native-cuda-v1"
FILES = (
    "model.safetensors",
    "base.safetensors",
    "behavior.safetensors",
    "training.pt",
    "actor.json",
    "last-frozen-epoch.json.gz",
)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


def _state(learner):
    return {
        "epoch": learner.epoch,
        "actor_steps": learner.actors.steps,
        "fresh_transitions": learner.actors.steps * learner.config.actors.games,
        "optimizer_accepted_updates": learner.optimizer_accepted_updates,
        "optimizer_attempted_updates": learner.optimizer_attempted_updates,
        "optimizer_rejected_updates": learner.optimizer_rejected_updates,
        "sample_chain_sha256": learner.sample_chain_sha256,
        "actors": learner.actors.cursor(),
        "replay_buffer": "closed-empty",
        "training_pass": "closed",
        "last_sampler_rng_state": learner.last_sampler_rng_state,
        "schedule_rng": learner.schedule_rng.getstate(),
        "search_rngs": [rng.getstate() for rng in learner.search_rngs],
        "pending_search_schedule": "closed-empty",
    }


def save_ownsearch_checkpoint(directory: Path, learner):
    learner.validate()
    if directory.exists():
        raise FileExistsError(directory)
    state = _state(learner)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        for name, model in zip(
            FILES[:3], (learner.online, learner.base, learner.behavior), strict=True
        ):
            save_weights(
                temp / name,
                model,
                provenance={
                    "source_commit": learner.source_commit,
                    "ownsearch_epoch": learner.epoch,
                },
            )
        numpy = np.random.get_state()
        torch.save(
            {
                "online": _cpu_tree(learner.online.state_dict()),
                "base": _cpu_tree(learner.base.state_dict()),
                "behavior": _cpu_tree(learner.behavior.state_dict()),
                "optimizer": _cpu_tree(learner.optimizer.state_dict()),
                "actor_rng": learner.actors.rng.getstate(),
                "sampler_seed_rng": learner.sampler_seed_rng.getstate(),
                "schedule_rng": learner.schedule_rng.getstate(),
                "search_rngs": [rng.getstate() for rng in learner.search_rngs],
                "last_sampler_rng_state": learner.last_sampler_rng_state,
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if learner.config.device == "cuda:0"
                else [],
                "numpy_rng": {
                    "kind": numpy[0],
                    "keys": torch.tensor(numpy[1].astype(np.int64)),
                    "position": numpy[2],
                    "has_gauss": numpy[3],
                    "cached_gauss": numpy[4],
                },
                "state": state,
            },
            temp / "training.pt",
        )
        (temp / "actor.json").write_text(
            json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (temp / "last-frozen-epoch.json.gz").write_bytes(learner.last_epoch_gzip)
        manifest = {
            "schema": CUDA_SCHEMA if learner.config.device == "cuda:0" else CPU_SCHEMA,
            "source_commit": learner.source_commit,
            "run_config": learner.run_config,
            "runtime": _runtime(learner.config.device),
            "state": state,
            "base_model_sha256": learner.base_model_sha256,
            "model_specification": learner.online.specification,
            "trainable": [
                n for n, p in learner.online.named_parameters() if p.requires_grad
            ],
            "model_modes": [
                m.training for m in (learner.online, learner.base, learner.behavior)
            ],
            "inputs": {
                k: {
                    "relative_path": os.path.relpath(p.resolve(), directory.resolve()),
                    "sha256": sha256(p),
                }
                for k, p in learner.input_paths.items()
            },
            "artifacts": {name: sha256(temp / name) for name in FILES},
        }
        (temp / "checkpoint.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        for file in temp.iterdir():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
        os.rename(temp, directory)
        descriptor = os.open(directory.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return manifest
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def load_ownsearch_checkpoint(directory: Path, learner):
    from harbichess.backends.torch_network import load_weights
    from harbichess.training.torch_fullgame_ppo import torch_model_digest
    from harbichess.training.torch_ownsearch_learner import (
        OWNSEARCH_BOARD_CACHE_SIZE,
        canonical,
    )

    manifest = json.loads((directory / "checkpoint.json").read_text())
    schema = CUDA_SCHEMA if learner.config.device == "cuda:0" else CPU_SCHEMA
    if (
        manifest["schema"] != schema
        or manifest["source_commit"] != learner.source_commit
        or manifest["run_config"] != learner.run_config
        or manifest["runtime"] != _runtime(learner.config.device)
        or manifest["model_specification"] != learner.online.specification
        or manifest["base_model_sha256"] != learner.base_model_sha256
        or manifest["trainable"]
        != [n for n, p in learner.online.named_parameters() if p.requires_grad]
        or set(manifest["artifacts"]) != set(FILES)
        or set(manifest["inputs"]) != set(learner.input_paths)
    ):
        raise ValueError("ownsearch native schema/source/config/runtime mismatch")
    for name, expected in manifest["artifacts"].items():
        path = directory / name
        if path.is_symlink() or sha256(path) != expected:
            raise ValueError("ownsearch native artifact integrity mismatch")
    for name, expected in manifest["inputs"].items():
        if (directory / expected["relative_path"]).resolve() != learner.input_paths[
            name
        ].resolve() or sha256(learner.input_paths[name]) != expected["sha256"]:
            raise ValueError("ownsearch native original input path/hash mismatch")
    training = torch.load(
        directory / "training.pt", map_location="cpu", weights_only=True
    )
    state = json.loads((directory / "actor.json").read_text())
    if (
        canonical(state) != canonical(training["state"])
        or state != manifest["state"]
        or state["replay_buffer"] != "closed-empty"
        or state["training_pass"] != "closed"
    ):
        raise ValueError("ownsearch native independent state mismatch")
    for field in (
        "epoch",
        "actor_steps",
        "fresh_transitions",
        "optimizer_accepted_updates",
        "optimizer_attempted_updates",
        "optimizer_rejected_updates",
    ):
        if type(state[field]) is not int or state[field] < 0:
            raise ValueError("ownsearch native invalid independent counters")
    if (
        state["actor_steps"] != state["epoch"] * learner.config.epoch_steps
        or state["fresh_transitions"]
        != state["actor_steps"] * learner.config.actors.games
        or state["actors"]["steps"] != state["actor_steps"]
    ):
        raise ValueError("ownsearch native collection counters differ")
    for name, model, filename in zip(
        ("online", "base", "behavior"),
        (learner.online, learner.base, learner.behavior),
        FILES[:3],
        strict=True,
    ):
        model.load_state_dict(training[name], strict=True)
        portable = load_weights(directory / filename)
        if torch_model_digest(model) != torch_model_digest(portable):
            raise ValueError("ownsearch native portable/native model bits differ")
    learner.optimizer.load_state_dict(training["optimizer"])
    learner.actors = OnlineActors(
        read_online_train_book(learner.input_paths["book"]),
        config=learner.config.actors,
        rng=random.Random(0),
        cursor=state["actors"],
    )
    from harbichess.chess.encoding import BoardEncoder

    learner.actors.rules.board_cache_size = OWNSEARCH_BOARD_CACHE_SIZE
    learner.encoder = BoardEncoder(learner.actors.rules)
    learner.actors.rng.setstate(training["actor_rng"])
    learner.sampler_seed_rng.setstate(training["sampler_seed_rng"])
    if (
        state["pending_search_schedule"] != "closed-empty"
        or len(training["search_rngs"]) != learner.config.actors.games
        or canonical(state["schedule_rng"]) != canonical(training["schedule_rng"])
        or canonical(state["search_rngs"]) != canonical(training["search_rngs"])
    ):
        raise ValueError("ownsearch closed schedule/search RNG inventory mismatch")
    learner.schedule_rng.setstate(training["schedule_rng"])
    for rng, saved in zip(learner.search_rngs, training["search_rngs"], strict=True):
        rng.setstate(saved)
    learner.pending_search_schedule = {}
    learner.last_sampler_rng_state = training["last_sampler_rng_state"]
    if canonical(learner.last_sampler_rng_state) != canonical(
        state["last_sampler_rng_state"]
    ):
        raise ValueError("ownsearch native sampler state differs")
    learner.epoch = state["epoch"]
    for field in (
        "optimizer_accepted_updates",
        "optimizer_attempted_updates",
        "optimizer_rejected_updates",
        "sample_chain_sha256",
    ):
        setattr(learner, field, state[field])
    learner.last_epoch_gzip = (directory / "last-frozen-epoch.json.gz").read_bytes()
    if learner.epoch:
        record = json.loads(gzip.decompress(learner.last_epoch_gzip))
        chain = record.pop("sample_chain_sha256")
        previous = record["previous_sample_chain_sha256"]
        if (
            record["schema"] != learner.run_config["schema"]
            or any(
                record[field] != state[field]
                for field in (
                    "epoch",
                    "actor_steps",
                    "optimizer_accepted_updates",
                    "optimizer_attempted_updates",
                    "optimizer_rejected_updates",
                )
            )
            or record["total_fresh_transitions"] != state["fresh_transitions"]
            or chain != learner.sample_chain_sha256
            or hashlib.sha256(bytes.fromhex(previous) + canonical(record)).hexdigest()
            != chain
        ):
            raise ValueError("ownsearch native last epoch/chain differs")
        epoch = deserialize_policy_epoch(canonical(record["collection"]))
        if (
            epoch.next_actor_cursor != learner.actors.cursor()
            or epoch.next_actor_rng_state != learner.actors.rng.getstate()
            or epoch.model_digest != torch_model_digest(learner.behavior)
        ):
            raise ValueError("ownsearch native epoch/behavior/actor cursors differ")
        targets = build_fullgame_targets(
            learner.actors.rules, epoch, claim_draw=learner.config.actors.claim_draw
        )
        from harbichess.selfplay.online_epoch import _tuplify
        from harbichess.training.ownsearch_targets import validate_search_ledger

        validate_search_ledger(
            epoch,
            record["own_search"],
            learner.config.search,
            rules=learner.actors.rules,
            claim_draw=learner.config.actors.claim_draw,
        )
        if (
            _tuplify(record["own_search"]["schedule_rng_after"])
            != learner.schedule_rng.getstate()
            or [_tuplify(s) for s in record["own_search"]["search_rng_after"]]
            != [r.getstate() for r in learner.search_rngs]
            or len(record["own_search"]["roots"])
            != record["training"]["policy_target_rows"]
        ):
            raise ValueError("ownsearch frozen target/schedule/search cursor mismatch")
        if canonical(record["training"]["sampler_rng_state"]) != canonical(
            learner.last_sampler_rng_state
        ):
            raise ValueError("ownsearch archived sampler state differs from native")
        if any(
            getattr(targets, name) != value
            for name, value in record["target_counts"].items()
        ):
            raise ValueError("ownsearch own-terminal/cap target counts differ")
        if len(targets.targets) != record["training"]["trained_transitions"]:
            raise ValueError("ownsearch native frozen complete target buffer differs")
    elif (
        learner.last_epoch_gzip
        or learner.sample_chain_sha256 != hashlib.sha256(b"").hexdigest()
        or learner.optimizer_accepted_updates
        or learner.optimizer_attempted_updates
    ):
        raise ValueError(
            "ownsearch initial native unexpectedly contains epoch/update state"
        )
    for model, mode in zip(
        (learner.online, learner.base, learner.behavior),
        manifest["model_modes"],
        strict=True,
    ):
        model.train(mode)
    learner.closed = True
    learner.validate()
    numpy = training["numpy_rng"]
    np.random.set_state(
        (
            numpy["kind"],
            numpy["keys"].numpy().astype(np.uint32),
            numpy["position"],
            numpy["has_gauss"],
            numpy["cached_gauss"],
        )
    )
    random.setstate(training["python_rng"])
    torch.set_rng_state(training["torch_rng"])
    cuda = training["cuda_rng"]
    if learner.config.device == "cuda:0":
        if len(cuda) != torch.cuda.device_count():
            raise ValueError("ownsearch native CUDA RNG device inventory differs")
        torch.cuda.set_rng_state_all(cuda)
    elif cuda:
        raise ValueError("CPU ownsearch native contains CUDA RNG")
    return manifest
