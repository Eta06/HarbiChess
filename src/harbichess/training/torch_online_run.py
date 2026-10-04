"""Explicit bounded online experiment runner; immutable journals and native resume.

Run only under a prospective protocol. The absolute deadline persists across
invocations; a planned process boundary is distinct from a failed time budget.
Legacy replay/checkpoints and weights-only imports are never overwritten.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harbichess.training.torch_online_learner import TorchOnlineConfig

_IMPORT_STARTED_EPOCH = time.time()


def _json(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )


def _publish_once(path, value):
    with path.open("xb") as handle:
        handle.write(_json(value))
        handle.flush()
        os.fsync(handle.fileno())


def _progress(directory, value):
    temporary = directory / ".progress.tmp"
    temporary.write_bytes(_json(value))
    temporary.replace(directory / "progress.json")


def _journal(path, record):
    content = _json(record)
    if path.exists():
        if gzip.decompress(path.read_bytes()) != content:
            raise ValueError("uncommitted online journal differs from deterministic regeneration")
        return
    compressed = gzip.compress(content, compresslevel=1, mtime=0)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    with temporary.open("xb") as handle:
        handle.write(compressed)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _verify_journals(directory, update, expected_chain):
    chain = hashlib.sha256(b"").hexdigest()
    for step in range(1, update + 1):
        path = directory / "journal" / f"update-{step:08d}.json.gz"
        record = json.loads(gzip.decompress(path.read_bytes()))
        current = record.pop("sample_chain_sha256")
        previous = record.pop("previous_sample_chain_sha256")
        if record["update"] != step or previous != chain:
            raise ValueError("online journal update/prefix differs")
        # Learner hashes canonical JSON without the final newline.
        chain = hashlib.sha256(bytes.fromhex(chain) + _json(record)[:-1]).hexdigest()
        if chain != current:
            raise ValueError("online journal content hash differs")
    if chain != expected_chain:
        raise ValueError("online journal chain differs from native checkpoint cursor")


def _process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]
    except FileNotFoundError:
        return None


def _acquire_lock(directory, source):
    path = directory / ".online-active.json"
    owner = {
        "pid": os.getpid(),
        "start_ticks": _process_start(os.getpid()),
        "source_commit": source,
    }
    for _ in range(2):
        try:
            _publish_once(path, owner)
            return path, owner
        except FileExistsError:
            old = json.loads(path.read_text())
            if old["source_commit"] != source or _process_start(old["pid"]) == old["start_ticks"]:
                raise ValueError(
                    "another live online invocation or differing source owns this run"
                ) from None
            # Remove only a proved dead/reused PID marker; never terminate a process.
            if json.loads(path.read_text()) != old:
                raise ValueError("online owner changed during stale-lock inspection") from None
            path.unlink()
    raise RuntimeError("could not acquire online run owner")


def _memory_metric():
    return "cgroup-v2-current" if Path("/sys/fs/cgroup/memory.current").is_file() \
        else "system-used-including-cache"


def _memory_usage(metric):
    if metric == "cgroup-v2-current":
        return int(Path("/sys/fs/cgroup/memory.current").read_text())
    if metric == "system-used-including-cache":
        values = {
            row.split(":")[0]: int(row.split()[1]) * 1024
            for row in Path("/proc/meminfo").read_text().splitlines()
            if row.startswith(("MemTotal:", "MemFree:"))
        }
        return values["MemTotal"] - values["MemFree"]
    raise ValueError("unknown registered memory metric")


def run_online(
    directory: Path,
    *,
    input_paths: dict[str, Path],
    config: TorchOnlineConfig,
    source_commit: str,
    max_updates: int,
    checkpoint_interval: int,
    deadline_epoch: float,
    stop_at: int | None = None,
    resume: Path | None = None,
    memory_max_bytes: int = 15 * 1024**3,
    disk_min_free_bytes: int = 8 * 1024**3,
) -> dict:
    started = time.perf_counter()
    cpu_started = time.process_time()
    from harbichess.backends.torch_network import sha256
    from harbichess.training.torch_online_learner import TorchOnlineLearner

    if (
        type(max_updates) is not int
        or max_updates <= 0
        or type(checkpoint_interval) is not int
        or checkpoint_interval <= 0
        or type(memory_max_bytes) is not int
        or memory_max_bytes <= 0
        or type(disk_min_free_bytes) is not int
        or disk_min_free_bytes <= 0
        or not math.isfinite(deadline_epoch)
        or deadline_epoch <= time.time()
        or (stop_at is not None and (type(stop_at) is not int or not 0 < stop_at <= max_updates))
    ):
        raise ValueError("invalid explicit online absolute limits/process boundary")
    actual_source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    checkout = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    if not Path(__file__).resolve().is_relative_to(checkout.resolve()):
        raise ValueError("online runner module is outside the declared source checkout")
    if actual_source != source_commit or subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ):
        raise ValueError("online invocation requires the declared clean source checkout")
    configuration = asdict(config)
    if config.device == "cpu":
        configuration.pop("device")
    memory_metric = _memory_metric()
    metadata = {
        "schema": 1,
        "source_commit": source_commit,
        "config": configuration,
        "inputs": {
            name: {
                "relative_path": os.path.relpath(path.resolve(), directory.resolve()),
                "sha256": sha256(path),
            }
            for name, path in input_paths.items()
        },
        "max_updates": max_updates,
        "checkpoint_interval": checkpoint_interval,
        "absolute_deadline_epoch": deadline_epoch,
        "memory_max_bytes": memory_max_bytes,
        "disk_min_free_bytes": disk_min_free_bytes,
        "scope": (
            "Fresh one-ply self-play transitions consumed once, one AdamW+EMA update per batch. "
            "No engine queries or automatic strength promotion."
        ),
    }
    # Original CPU/cgroup-v2 metadata stays byte-compatible.
    if config.device != "cpu" or memory_metric != "cgroup-v2-current":
        metadata["memory_metric"] = memory_metric
    if resume is None:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "journal").mkdir()
        (directory / "checkpoints").mkdir()
        _publish_once(directory / "metadata.json", metadata)
    else:
        if resume.resolve().parent.parent != directory.resolve():
            raise ValueError("native online checkpoint must belong to this run")
        if json.loads((directory / "metadata.json").read_text()) != metadata:
            raise ValueError(
                "online resumed source/input/config/budget changed; deadline extension refused"
            )
    invocation = str(time.time_ns())
    _publish_once(
        directory / f"invocation-{invocation}-command.json",
        {
            "source_commit": source_commit,
            "resume": str(resume) if resume else None,
            "stop_at": stop_at,
            "started_epoch": time.time(),
            "module_import_started_epoch": _IMPORT_STARTED_EPOCH,
            "remaining_whole_seconds": deadline_epoch - time.time(),
        },
    )
    lock, owner = _acquire_lock(directory, source_commit)
    learner = None
    status, reason, error = "failed", "initialization failed", None
    try:

        def guard():
            if time.time() >= deadline_epoch:
                raise TimeoutError("registered absolute online deadline exhausted; no extension")
            if _memory_usage(memory_metric) > metadata["memory_max_bytes"]:
                raise RuntimeError("online registered memory ceiling exceeded")
            if shutil.disk_usage(directory).free < metadata["disk_min_free_bytes"]:
                raise RuntimeError("online registered disk free floor violated")

        guard()
        if resume is None:
            learner = TorchOnlineLearner.fresh(
                config=config, input_paths=input_paths, source_commit=source_commit
            )
            guard()
            learner.checkpoint(directory / "checkpoints/step-00000000")
        else:
            learner = TorchOnlineLearner.resume(
                resume, config=config, input_paths=input_paths, source_commit=source_commit
            )
            _verify_journals(directory, learner.update, learner.sample_chain_sha256)
        if learner.update >= max_updates or (stop_at is not None and stop_at <= learner.update):
            raise ValueError("online requested boundary/target already reached")
        while learner.update < max_updates:
            guard()
            record = learner.train_update()
            _journal(directory / "journal" / f"update-{learner.update:08d}.json.gz", record)
            boundary = stop_at == learner.update
            final = learner.update == max_updates
            if boundary or final or learner.update % checkpoint_interval == 0:
                guard()
                learner.checkpoint(directory / "checkpoints" / f"step-{learner.update:08d}")
            _progress(
                directory,
                {
                    "status": "running",
                    "update": learner.update,
                    "transitions": learner.update * config.actors.games,
                    "remaining_whole_seconds": deadline_epoch - time.time(),
                    "sample_chain_sha256": learner.sample_chain_sha256,
                    "loss": record["loss"],
                    "terminated_games": record["terminated_games"],
                },
            )
            if boundary or final:
                guard()
                status, reason = (
                    "completed",
                    "registered process boundary" if boundary else "maximum updates",
                )
                break
    except Exception as caught:
        error, reason = repr(caught), str(caught)
    finally:
        if lock.exists() and json.loads(lock.read_text()) == owner:
            lock.unlink()
    result = {
        "status": status,
        "reason": reason,
        "error": error,
        "source_commit": source_commit,
        "update": learner.update if learner else None,
        "transitions": learner.update * config.actors.games if learner else None,
        "sample_chain_sha256": learner.sample_chain_sha256 if learner else None,
        "run_function_wall_seconds": time.perf_counter() - started,
        "run_function_cpu_seconds": time.process_time() - cpu_started,
        "module_import_started_epoch": _IMPORT_STARTED_EPOCH,
        "finished_epoch": time.time(),
        "absolute_deadline_epoch": deadline_epoch,
        "scope": (
            "Infrastructure/training receipt; no strength, retention or model promotion conclusion."
        ),
    }
    _publish_once(directory / f"invocation-{invocation}-result.json", result)
    _progress(directory, result)
    print(json.dumps(result), flush=True)
    if status != "completed":
        raise RuntimeError(reason)
    return result


def main():
    from harbichess.selfplay.online_actor import OnlineActorConfig
    from harbichess.training.online_objective import OnlineObjectiveConfig
    from harbichess.training.torch_online_learner import TorchOnlineConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-updates", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--memory-max-bytes", type=int, default=15 * 1024**3)
    parser.add_argument("--disk-min-free-bytes", type=int, default=8 * 1024**3)
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text())
    configuration["actors"] = OnlineActorConfig(**configuration["actors"])
    configuration["objective"] = OnlineObjectiveConfig(**configuration["objective"])
    run_online(
        args.directory,
        input_paths={
            "initial_weights": args.weights,
            "book": args.book,
            "experiment_config": args.config,
            "protocol": args.protocol,
        },
        config=TorchOnlineConfig(**configuration),
        memory_max_bytes=args.memory_max_bytes,
        disk_min_free_bytes=args.disk_min_free_bytes,
        source_commit=args.source_commit,
        max_updates=args.max_updates,
        checkpoint_interval=args.checkpoint_interval,
        deadline_epoch=args.deadline_epoch,
        stop_at=args.stop_at,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
