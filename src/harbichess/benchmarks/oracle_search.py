"""Benchmark oracle-process allocation in the qualified KOPRU search workload."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import mlx.core as mx

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.mcts import MCTS, SearchConfig
from harbichess.search.value_oracle import (
    OracleValueEvaluator,
    ProcessTacticalOracle,
    TacticalOracleConfig,
)
from harbichess.selfplay.parallel import run_parallel_searches


@dataclass(frozen=True, slots=True)
class OracleSearchBenchmarkConfig:
    checkpoint: Path
    replay_shard: Path
    output: Path
    oracle_workers: tuple[int, ...] = (6, 8, 10, 12, 14)
    positions: int = 24
    simulations: int = 64
    repeats: int = 2
    actor_workers: int = 24
    max_batch_size: int = 48
    max_wait_seconds: float = 0.00025
    oracle_depth: int = 1

    def __post_init__(self) -> None:
        if (
            not self.oracle_workers
            or min(self.oracle_workers) <= 0
            or min(
                self.positions,
                self.simulations,
                self.repeats,
                self.actor_workers,
                self.max_batch_size,
                self.oracle_depth,
            )
            <= 0
            or self.max_wait_seconds < 0
        ):
            raise ValueError("oracle search benchmark configuration is invalid")


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _stratified_states(config: OracleSearchBenchmarkConfig):
    records = read_shard(config.replay_shard).records
    if len(records) < config.positions:
        raise ValueError("benchmark replay has fewer records than requested positions")
    return tuple(
        records[index * len(records) // config.positions].state for index in range(config.positions)
    )


def run_oracle_search_benchmark(config: OracleSearchBenchmarkConfig) -> Path:
    if config.output.exists():
        raise FileExistsError(f"benchmark output already exists: {config.output}")
    rules = PythonChessRules()
    states = _stratified_states(config)
    network_config = NetworkConfig(
        trunk_channels=16,
        residual_blocks=2,
        policy_channels=4,
        value_channels=2,
        value_hidden=32,
    )
    network = HarbiChessNetwork(network_config)
    network.load_weights(str(config.checkpoint))
    rows = []
    with SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=config.max_batch_size,
        max_wait_seconds=config.max_wait_seconds,
    ) as batcher:
        neural = NeuralPositionEvaluator(batcher, rules=rules)
        for worker_count in config.oracle_workers:
            oracle = ProcessTacticalOracle(
                TacticalOracleConfig(depth=config.oracle_depth), workers=worker_count
            )
            try:
                with ThreadPoolExecutor(max_workers=config.actor_workers) as pool:
                    tuple(pool.map(oracle.value, states))
                search = MCTS(
                    OracleValueEvaluator(neural, oracle),
                    rules=rules,
                    config=SearchConfig(simulations=config.simulations),
                )
                repetitions = []
                for repeat in range(config.repeats):
                    batcher.reset_statistics()
                    started = time.perf_counter()
                    run_parallel_searches(
                        search,
                        states,
                        tuple(
                            repeat * config.positions + index for index in range(config.positions)
                        ),
                        max_workers=config.actor_workers,
                        temperature=1.0,
                    )
                    elapsed = time.perf_counter() - started
                    statistics = batcher.reset_statistics()
                    repetitions.append(
                        {
                            "elapsed_seconds": elapsed,
                            "simulations_per_second": (
                                config.positions * config.simulations / elapsed
                            ),
                            "backend_positions": statistics.positions,
                            "backend_batches": statistics.batches,
                            "average_batch_size": statistics.average_batch_size,
                            "largest_batch": statistics.largest_batch,
                            "backend_seconds": statistics.backend_seconds,
                            "average_queue_wait_ms": statistics.average_queue_wait_ms,
                        }
                    )
                rows.append(
                    {
                        "oracle_workers": worker_count,
                        "mean_elapsed_seconds": mean(row["elapsed_seconds"] for row in repetitions),
                        "mean_simulations_per_second": mean(
                            row["simulations_per_second"] for row in repetitions
                        ),
                        "mean_batch_size": mean(row["average_batch_size"] for row in repetitions),
                        "repeats": repetitions,
                    }
                )
            finally:
                oracle.close()
    payload = {
        "source_commit": _source_commit(),
        "created_at": time.time(),
        "platform": platform.platform(),
        "mlx_device": mx.device_info(),
        "config": {
            **asdict(config),
            "checkpoint": str(config.checkpoint),
            "replay_shard": str(config.replay_shard),
            "output": str(config.output),
        },
        "results": rows,
        "best_oracle_workers": max(rows, key=lambda row: row["mean_simulations_per_second"])[
            "oracle_workers"
        ],
    }
    _atomic_json(config.output, payload)
    return config.output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--replay-shard", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--oracle-workers", default="6,8,10,12,14")
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--actor-workers", type=int, default=24)
    parser.add_argument("--max-batch-size", type=int, default=48)
    parser.add_argument("--max-wait-ms", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = run_oracle_search_benchmark(
        OracleSearchBenchmarkConfig(
            checkpoint=arguments.checkpoint,
            replay_shard=arguments.replay_shard,
            output=arguments.output,
            oracle_workers=tuple(int(value) for value in arguments.oracle_workers.split(",")),
            positions=arguments.positions,
            simulations=arguments.simulations,
            repeats=arguments.repeats,
            actor_workers=arguments.actor_workers,
            max_batch_size=arguments.max_batch_size,
            max_wait_seconds=arguments.max_wait_ms / 1_000,
        )
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
