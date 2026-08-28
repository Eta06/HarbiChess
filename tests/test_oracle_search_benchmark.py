from pathlib import Path

import pytest

benchmark_module = pytest.importorskip("harbichess.benchmarks.oracle_search")
OracleSearchBenchmarkConfig = benchmark_module.OracleSearchBenchmarkConfig
_parser = benchmark_module._parser


def test_oracle_search_benchmark_validates_worker_allocation() -> None:
    with pytest.raises(ValueError, match="configuration is invalid"):
        OracleSearchBenchmarkConfig(
            checkpoint=Path("model.safetensors"),
            replay_shard=Path("validation.jsonl.gz"),
            output=Path("result.json"),
            oracle_workers=(0,),
        )


def test_oracle_search_benchmark_accepts_frozen_workload() -> None:
    config = OracleSearchBenchmarkConfig(
        checkpoint=Path("model.safetensors"),
        replay_shard=Path("validation.jsonl.gz"),
        output=Path("result.json"),
        oracle_workers=(8, 12),
        positions=24,
        simulations=64,
        repeats=2,
    )

    assert config.oracle_workers == (8, 12)
    assert config.actor_workers == 24


def test_oracle_search_benchmark_exposes_batch_scheduling_controls() -> None:
    arguments = _parser().parse_args(
        [
            "--checkpoint",
            "model.safetensors",
            "--replay-shard",
            "validation.jsonl.gz",
            "--output",
            "result.json",
            "--actor-workers",
            "32",
            "--max-batch-size",
            "64",
            "--max-wait-ms",
            "0.1",
        ]
    )

    assert arguments.actor_workers == 32
    assert arguments.max_batch_size == 64
    assert arguments.max_wait_ms == pytest.approx(0.1)
