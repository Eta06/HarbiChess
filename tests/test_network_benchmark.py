import pytest

mx = pytest.importorskip("mlx.core")
benchmark_module = pytest.importorskip("harbichess.benchmarks.network")
network_module = pytest.importorskip("harbichess.backends.mlx_network")


def test_real_network_benchmark_reports_batch_throughput() -> None:
    result = benchmark_module.run_benchmark(
        network_module.NetworkConfig(trunk_channels=8, residual_blocks=1),
        batch_sizes=(1, 2),
        iterations=1,
        dtype=mx.float32,
        compiled=False,
    )

    assert result.parameters > 0
    assert [batch.batch_size for batch in result.batches] == [1, 2]
    assert all(batch.positions_per_second > 0 for batch in result.batches)


def test_benchmark_rejects_invalid_workload() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        benchmark_module.run_benchmark(
            network_module.NetworkConfig(trunk_channels=8, residual_blocks=1),
            batch_sizes=(0,),
            iterations=1,
        )
