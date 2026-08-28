from pathlib import Path

import pytest

from harbichess.benchmarks.full_gumbel_batching import (
    FullGumbelBatchBenchmarkConfig,
    _compare,
)


def _row() -> dict[str, object]:
    return {
        "identity": "game:0:0",
        "selected_action": "a2a3",
        "root_visits": (("a2a3", 4),),
        "root_value": 0.25,
        "target": (("a2a3", 1.0),),
    }


def test_batch_benchmark_equivalence_requires_exact_search_output() -> None:
    row = _row()
    equivalent = _compare((row,), (dict(row),))
    changed = _compare((row,), ({**row, "root_value": 0.2},))

    assert equivalent["passed"] is True
    assert equivalent["maximum_output_delta"] == 0.0
    assert changed["passed"] is False


def test_batch_benchmark_rejects_duplicate_wait_windows() -> None:
    with pytest.raises(ValueError, match="wait-window"):
        FullGumbelBatchBenchmarkConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            train_shard=Path("train"),
            target_result=Path("target"),
            wait_windows=(0.001, 0.001),
        )
