import pytest

from harbichess.chess.actions import POLICY_SIZE
from harbichess.core.backend import BackendCapabilities, PolicyValueOutput

benchmark_module = pytest.importorskip("harbichess.benchmarks.search")
benchmark_parallel_searches = benchmark_module.benchmark_parallel_searches


class UniformBackend:
    capabilities = BackendCapabilities("uniform", "cpu", False, False)

    def evaluate(self, positions):
        output = PolicyValueOutput((0.0,) * POLICY_SIZE, (0.0, 0.0, 0.0))
        return [output] * len(positions)


def test_parallel_search_benchmark_reports_batch_utilization() -> None:
    results = benchmark_parallel_searches(
        UniformBackend(),
        game_counts=[1, 2],
        simulations=2,
        max_batch_size=2,
        max_wait_seconds=0.01,
    )

    assert [result.games for result in results] == [1, 2]
    assert all(result.backend_positions > 0 for result in results)
    assert results[1].largest_batch == 2
    assert results[1].simulations_per_second > 0


def test_parallel_search_benchmark_validates_workload() -> None:
    with pytest.raises(ValueError, match="positive"):
        benchmark_parallel_searches(UniformBackend(), game_counts=[0], simulations=1)
