"""Small MLX device benchmark used to verify the local execution path."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DeviceBenchmark:
    device: str
    size: int
    iterations: int
    elapsed_seconds: float
    operations_per_second: float


def benchmark_device(device: str, *, size: int, iterations: int) -> DeviceBenchmark:
    if size <= 0 or iterations <= 0:
        raise ValueError("size and iterations must be positive")

    import mlx.core as mx

    target = mx.gpu if device == "gpu" else mx.cpu
    left = mx.random.uniform(shape=(size, size))
    right = mx.random.uniform(shape=(size, size))

    with mx.stream(target):
        mx.eval(left @ right)
        mx.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            mx.eval(left @ right)
        mx.synchronize()
    elapsed = time.perf_counter() - started
    operation_count = 2 * size**3 * iterations
    return DeviceBenchmark(
        device=device,
        size=size,
        iterations=iterations,
        elapsed_seconds=elapsed,
        operations_per_second=operation_count / elapsed,
    )


def run_benchmarks(*, size: int, iterations: int) -> list[DeviceBenchmark]:
    return [
        benchmark_device(device, size=size, iterations=iterations)
        for device in ("cpu", "gpu")
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=5)
    arguments = parser.parse_args(argv)
    results = run_benchmarks(size=arguments.size, iterations=arguments.iterations)
    print(json.dumps([asdict(result) for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
