"""Bounded own-search CE and own-terminal WDL runner with a distinct native schema."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from harbichess.backends.torch_network import sha256
from harbichess.selfplay.online_actor import OnlineActorConfig
from harbichess.training.ownsearch_targets import OwnSearchConfig
from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
from harbichess.training.torch_online_run import (
    _acquire_lock,
    _memory_metric,
    _memory_usage,
    _progress,
    _publish_once,
)
from harbichess.training.torch_ownsearch_core import OwnSearchObjective
from harbichess.training.torch_search_acting_learner import (
    EMPTY_CHAIN,
    TorchSearchActingConfig,
    TorchSearchActingLearner,
    canonical,
)


def clean_source(source):
    checkout = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip() != source or subprocess.check_output(["git", "status", "--porcelain"], text=True):
        raise ValueError("fullgame invocation requires declared clean source checkout")
    for name, module in sys.modules.copy().items():
        if (
            name.startswith("harbichess.")
            and getattr(module, "__file__", None)
            and not Path(module.__file__).resolve().is_relative_to(checkout)
        ):
            raise ValueError("fullgame imported HarbiChess module outside pinned checkout")
    return checkout


def publish_bytes(path, data):
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError("existing fullgame immutable artifact differs")
        return
    temp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def verify_epochs(directory, epoch, chain):
    current = EMPTY_CHAIN
    attempted = accepted = discarded = 0
    for i in range(1, epoch + 1):
        path = directory / "journal" / f"epoch-{i:08d}.json.gz"
        record = json.loads(gzip.decompress(path.read_bytes()))
        expected = record.pop("sample_chain_sha256")
        if record["epoch"] != i or record["previous_sample_chain_sha256"] != current:
            raise ValueError("fullgame journal contiguous epoch/chain differs")
        attempted += record["training"]["optimizer_steps_attempted"]
        accepted += record["training"]["optimizer_steps_committed"]
        discarded = attempted - accepted
        if (
            record["optimizer_attempted_updates"] != attempted
            or record["optimizer_accepted_updates"] != accepted
            or record["optimizer_rejected_updates"] != discarded
        ):
            raise ValueError("fullgame journal independent optimizer prefix counters differ")
        current = hashlib.sha256(bytes.fromhex(current) + canonical(record)).hexdigest()
        if expected != current:
            raise ValueError("fullgame immutable journal hash differs")
    if current != chain:
        raise ValueError("fullgame native and journal prefix chain differ")


def run_search_acting(
    directory,
    *,
    config,
    input_paths,
    source_commit,
    max_epochs,
    checkpoint_interval,
    deadline_epoch,
    memory_max_bytes,
    disk_min_free_bytes,
    stop_at=None,
    resume=None,
    memory_policy="legacy-total-v1",
):
    if (
        type(max_epochs) is not int
        or max_epochs <= 0
        or type(checkpoint_interval) is not int
        or checkpoint_interval <= 0
        or type(memory_max_bytes) is not int
        or memory_max_bytes <= 0
        or type(disk_min_free_bytes) is not int
        or disk_min_free_bytes <= 0
        or not math.isfinite(deadline_epoch)
        or deadline_epoch <= time.time()
        or (stop_at is not None and (type(stop_at) is not int or not 0 < stop_at <= max_epochs))
    ):
        raise ValueError("fullgame requires explicit whole-run deadline/resources/boundaries")
    if memory_policy not in ("legacy-total-v1", "inactive-file-v1"):
        raise ValueError("unknown registered memory policy")
    clean_source(source_commit)
    memory_budget = None
    if memory_policy == "inactive-file-v1":
        from harbichess.training.cgroup_budget import CgroupMemoryBudget

        memory_budget = CgroupMemoryBudget(memory_max_bytes)
    metric = _memory_metric()
    metadata = {
        "schema": "search-acting-supervised-run-v2",
        "config": asdict(config),
        "source_commit": source_commit,
        "inputs": {
            k: {
                "relative_path": os.path.relpath(p.resolve(), directory.resolve()),
                "sha256": sha256(p),
            }
            for k, p in input_paths.items()
        },
        "max_epochs": max_epochs,
        "checkpoint_interval": checkpoint_interval,
        "absolute_deadline_epoch": deadline_epoch,
        "memory_max_bytes": memory_max_bytes,
        "disk_min_free_bytes": disk_min_free_bytes,
        "memory_metric": metric,
        "scope": (
            "Own-model sparse Gumbel CE/search-acting actors + completed own-terminal WDL; "
            "no promotion."
        ),
    }
    if memory_budget is not None:
        metadata["memory_policy"] = memory_policy
        metadata["effective_active_budget_bytes"] = memory_budget.budget_bytes
        metadata["memory_metric"] = "cgroup-inactive-file-budget-v1"
    if resume is None:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "journal").mkdir()
        (directory / "checkpoints").mkdir()
        _publish_once(directory / "metadata.json", metadata)
    elif (
        resume.resolve().parent.parent != directory.resolve()
        or json.loads((directory / "metadata.json").read_text()) != metadata
    ):
        raise ValueError("fullgame resume source/input/config/resource/deadline changed")
    invocation = str(time.time_ns())
    _publish_once(
        directory / f"invocation-{invocation}-command.json",
        {
            "source_commit": source_commit,
            "resume": str(resume) if resume else None,
            "stop_at": stop_at,
            "started_epoch": time.time(),
            "absolute_deadline_epoch": deadline_epoch,
        },
    )
    lock, owner = _acquire_lock(directory, source_commit)
    learner = None
    started = time.perf_counter()
    result = {"status": "failed", "reason": "initialization failed"}
    try:

        def guard():
            if time.time() >= deadline_epoch:
                raise TimeoutError("whole fullgame absolute deadline exhausted; no extension")
            if memory_budget is not None:
                memory_budget.check()
            elif _memory_usage(metric) > memory_max_bytes:
                raise RuntimeError("registered fullgame memory ceiling exceeded")
            if shutil.disk_usage(directory).free < disk_min_free_bytes:
                raise RuntimeError("registered fullgame disk free floor violated")

        guard()
        if resume is None:
            learner = TorchSearchActingLearner.fresh(
                config=config, input_paths=input_paths, source_commit=source_commit
            )
            guard()
            learner.checkpoint(directory / "checkpoints/epoch-00000000")
        else:
            learner = TorchSearchActingLearner.resume(
                resume,
                config=config,
                input_paths=input_paths,
                source_commit=source_commit,
            )
            verify_epochs(directory, learner.epoch, learner.sample_chain_sha256)
            if (
                learner.epoch
                and (directory / "journal" / f"epoch-{learner.epoch:08d}.json.gz").read_bytes()
                != learner.last_epoch_gzip
            ):
                raise ValueError(
                    "last fullgame native frozen epoch bytes differ from published journal"
                )
        if learner.epoch >= max_epochs or (stop_at is not None and stop_at <= learner.epoch):
            raise ValueError("requested fullgame boundary already reached")
        while learner.epoch < max_epochs:
            guard()
            learner.train_epoch(guard=guard)
            publish_bytes(
                directory / "journal" / f"epoch-{learner.epoch:08d}.json.gz",
                learner.last_epoch_gzip,
            )
            boundary = learner.epoch == stop_at
            final = learner.epoch == max_epochs
            if boundary or final or learner.epoch % checkpoint_interval == 0:
                guard()
                path = directory / "checkpoints" / f"epoch-{learner.epoch:08d}"
                learner.checkpoint(path)
            _progress(
                directory,
                {
                    "status": "running",
                    "epoch": learner.epoch,
                    "fresh_transitions": learner.actors.steps * config.actors.games,
                    "optimizer_accepted_updates": learner.optimizer_accepted_updates,
                    "optimizer_attempted_updates": learner.optimizer_attempted_updates,
                    "optimizer_rejected_updates": learner.optimizer_rejected_updates,
                    "sample_chain_sha256": learner.sample_chain_sha256,
                    "remaining_whole_seconds": deadline_epoch - time.time(),
                },
            )
            if boundary or final:
                guard()
                result = {
                    "status": "completed",
                    "reason": "registered process boundary" if boundary else "maximum epochs",
                }
                break
    except Exception as caught:
        result = {"status": "failed", "reason": str(caught), "error": repr(caught)}
        if memory_budget is not None:
            result["memory_snapshot"] = memory_budget.last_snapshot
    finally:
        if lock.exists() and json.loads(lock.read_text()) == owner:
            lock.unlink()
    result |= {
        "schema": "search-acting-supervised-invocation-v2",
        "source_commit": source_commit,
        "epoch": learner.epoch if learner else None,
        "actor_steps": learner.actors.steps if learner else None,
        "optimizer_accepted_updates": learner.optimizer_accepted_updates if learner else None,
        "optimizer_attempted_updates": learner.optimizer_attempted_updates if learner else None,
        "optimizer_rejected_updates": learner.optimizer_rejected_updates if learner else None,
        "closed_boundary": learner.closed if learner else None,
        "sample_chain_sha256": learner.sample_chain_sha256 if learner else None,
        "absolute_deadline_epoch": deadline_epoch,
        "finished_epoch": time.time(),
        "wall_seconds": time.perf_counter() - started,
    }
    if memory_budget is not None:
        result.setdefault("memory_snapshot", memory_budget.last_snapshot)
    _publish_once(directory / f"invocation-{invocation}-result.json", result)
    _progress(directory, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "completed":
        raise RuntimeError(result["reason"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    for name in ("weights", "book", "config", "protocol"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--memory-max-bytes", type=int, required=True)
    parser.add_argument("--disk-min-free-bytes", type=int, required=True)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--memory-policy",
        choices=("legacy-total-v1", "inactive-file-v1"),
        default="legacy-total-v1",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["actors"] = OnlineActorConfig(**config["actors"])
    config["objective"] = OwnSearchObjective(**config["objective"])
    config["schedule"] = FullGamePPOTrainConfig(**config["schedule"])
    config["search"] = OwnSearchConfig(**config["search"])
    run_search_acting(
        args.directory,
        config=TorchSearchActingConfig(**config),
        input_paths={
            "initial_weights": args.weights,
            "book": args.book,
            "experiment_config": args.config,
            "protocol": args.protocol,
        },
        source_commit=args.source_commit,
        max_epochs=args.max_epochs,
        checkpoint_interval=args.checkpoint_interval,
        deadline_epoch=args.deadline_epoch,
        memory_max_bytes=args.memory_max_bytes,
        disk_min_free_bytes=args.disk_min_free_bytes,
        stop_at=args.stop_at,
        resume=args.resume,
        memory_policy=args.memory_policy,
    )


if __name__ == "__main__":
    main()
