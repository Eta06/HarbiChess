"""Benchmark the real HarbiChess policy/WDL network across inference batches."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import mlx.core as mx

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.encoding import ENCODER_CHANNELS


@dataclass(frozen=True, slots=True)
class BatchResult:
    batch_size: int
    latency_ms: float
    positions_per_second: float


@dataclass(frozen=True, slots=True)
class NetworkBenchmark:
    device: str
    dtype: str
    parameters: int
    config: dict[str, int]
    batches: list[BatchResult]


def benchmark_batch(
    forward: Callable[[mx.array], tuple[mx.array, mx.array]],
    *,
    batch_size: int,
    iterations: int,
    dtype: mx.Dtype,
) -> BatchResult:
    inputs = mx.random.uniform(shape=(batch_size, 8, 8, ENCODER_CHANNELS)).astype(dtype)
    mx.eval(inputs)
    for _ in range(3):
        mx.eval(*forward(inputs))
    mx.synchronize()

    started = time.perf_counter()
    for _ in range(iterations):
        mx.eval(*forward(inputs))
    mx.synchronize()
    elapsed = time.perf_counter() - started
    latency = elapsed / iterations
    return BatchResult(
        batch_size=batch_size,
        latency_ms=latency * 1_000,
        positions_per_second=batch_size / latency,
    )


def run_benchmark(
    config: NetworkConfig,
    *,
    batch_sizes: Sequence[int],
    iterations: int,
    dtype: mx.Dtype = mx.bfloat16,
    compiled: bool = True,
) -> NetworkBenchmark:
    if iterations <= 0 or any(batch <= 0 for batch in batch_sizes):
        raise ValueError("iterations and batch sizes must be positive")
    model = HarbiChessNetwork(config)
    model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    forward = mx.compile(model) if compiled else model
    results = [
        benchmark_batch(
            forward,
            batch_size=batch_size,
            iterations=iterations,
            dtype=dtype,
        )
        for batch_size in batch_sizes
    ]
    return NetworkBenchmark(
        device=mx.device_info()["device_name"],
        dtype=str(dtype),
        parameters=model.parameter_count,
        config={
            "input_channels": config.input_channels,
            "trunk_channels": config.trunk_channels,
            "residual_blocks": config.residual_blocks,
            "policy_size": config.policy_size,
        },
        batches=results,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,8,16,32,64,128")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--no-compile", action="store_true")
    arguments = parser.parse_args(argv)
    batch_sizes = tuple(int(value) for value in arguments.batches.split(","))
    config = NetworkConfig(
        trunk_channels=arguments.channels,
        residual_blocks=arguments.blocks,
    )
    result = run_benchmark(
        config,
        batch_sizes=batch_sizes,
        iterations=arguments.iterations,
        compiled=not arguments.no_compile,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

