"""Benchmark parallel PUCT searches through the shared MLX inference queue."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import mlx.core as mx

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.backend import PolicyValueBackend
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.mcts import MCTS, SearchConfig
from harbichess.selfplay.parallel import run_parallel_searches


@dataclass(frozen=True, slots=True)
class ParallelSearchResult:
    games: int
    simulations_per_game: int
    elapsed_seconds: float
    simulations_per_second: float
    backend_positions: int
    backend_batches: int
    average_batch_size: float
    largest_batch: int


def benchmark_parallel_searches(
    backend: PolicyValueBackend,
    *,
    game_counts: Sequence[int],
    simulations: int,
    max_batch_size: int = 128,
    max_wait_seconds: float = 0.001,
) -> list[ParallelSearchResult]:
    if simulations <= 0 or any(games <= 0 for games in game_counts):
        raise ValueError("simulations and game counts must be positive")
    rules = PythonChessRules()
    initial_state = rules.initial_state()
    encoded_initial = BoardEncoder(rules).encode(initial_state)
    results = []
    with SharedBatchEvaluator(
        backend,
        max_batch_size=max_batch_size,
        max_wait_seconds=max_wait_seconds,
    ) as batches:
        batches.evaluate(encoded_initial)
        batches.reset_statistics()
        search = MCTS(
            NeuralPositionEvaluator(batches, rules=rules),
            rules=rules,
            config=SearchConfig(simulations=simulations),
        )
        for games in game_counts:
            started = time.perf_counter()
            run_parallel_searches(
                search,
                [initial_state] * games,
                list(range(1, games + 1)),
                max_workers=games,
                temperature=1.0,
            )
            elapsed = time.perf_counter() - started
            statistics = batches.reset_statistics()
            results.append(
                ParallelSearchResult(
                    games=games,
                    simulations_per_game=simulations,
                    elapsed_seconds=elapsed,
                    simulations_per_second=games * simulations / elapsed,
                    backend_positions=statistics.positions,
                    backend_batches=statistics.batches,
                    average_batch_size=statistics.average_batch_size,
                    largest_batch=statistics.largest_batch,
                )
            )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="1,2,4,8,16")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--wait-ms", type=float, default=1.0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--no-compile", action="store_true")
    arguments = parser.parse_args(argv)

    network = HarbiChessNetwork(
        NetworkConfig(
            trunk_channels=arguments.channels,
            residual_blocks=arguments.blocks,
        )
    )
    backend = MLXPolicyValueBackend(
        network,
        dtype=mx.bfloat16,
        compiled=not arguments.no_compile,
    )
    results = benchmark_parallel_searches(
        backend,
        game_counts=tuple(int(value) for value in arguments.games.split(",")),
        simulations=arguments.simulations,
        max_batch_size=arguments.max_batch,
        max_wait_seconds=arguments.wait_ms / 1_000,
    )
    print(json.dumps([asdict(result) for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
