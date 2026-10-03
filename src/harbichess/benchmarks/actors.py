"""Startup-inclusive, real legal-game CPU actor throughput comparison."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from harbichess.backends.torch_network import load_weights, sha256
from harbichess.chess.rules import PythonChessRules
from harbichess.replay.schema import records_from_game
from harbichess.replay.shard import ShardMetadata, write_shard_atomic
from harbichess.replay.split import ReplaySplit
from harbichess.selfplay.torch_actors import collect_process_games, collect_thread_games
from harbichess.training.oracle_data import publish_json
from harbichess.training.torch_loop import LoopConfig


def run(weights: Path, directory: Path, mode: str, *, wall_seconds: float = 120) -> dict:
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = asdict(LoopConfig(seed=20261004, games=8, simulations=16, max_plies=64, workers=4))
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    started, cpu_start = time.perf_counter(), time.process_time()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    if mode == "thread":
        games, stats = collect_thread_games(
            load_weights(weights), config, range(8), started + wall_seconds
        )
    elif mode == "process":
        games, stats = collect_process_games(weights, config, range(8), started + wall_seconds)
    else:
        raise ValueError("unknown CPU actor mode")
    elapsed = time.perf_counter() - started
    rules = PythonChessRules()
    records = tuple(
        r
        for game in games
        for r in records_from_game(game, run_id="ayna-process-speed", rules=rules)
    )
    path = directory / "replay.jsonl.gz"
    write_shard_atomic(
        path,
        records,
        ShardMetadata(
            "ayna-process-speed",
            0,
            sha256(weights),
            source,
            datetime.now(UTC).isoformat(),
            ReplaySplit.TRAIN,
        ),
    )
    child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "status": "completed",
        "source_commit": source,
        "mode": mode,
        "weights_sha256": sha256(weights),
        "config": config,
        "games": len(games),
        "generated_positions": len(records),
        "wall_seconds": elapsed,
        "positions_per_wall_second": len(records) / elapsed,
        "process_cpu_seconds": time.process_time() - cpu_start,
        "children_cpu_seconds": child_end.ru_utime
        + child_end.ru_stime
        - child_start.ru_utime
        - child_start.ru_stime,
        "self_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "statistics": stats,
        "replay_sha256": sha256(path),
        "outcomes": [game.outcome.termination for game in games],
        "scope": "CPU throughput only, capped outcomes unknown, no strength claim",
    }
    publish_json(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mode", choices=("thread", "process"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.weights, args.directory, args.mode)), flush=True)


if __name__ == "__main__":
    main()
