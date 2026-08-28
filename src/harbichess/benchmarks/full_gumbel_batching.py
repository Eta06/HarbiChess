"""Frozen wait-window benchmark for fixed-shape Full Gumbel inference."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import RunMode, SnapshotStore
from harbichess.evaluation.full_gumbel_targets import _determinism_delta, _target_row
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS


@dataclass(frozen=True, slots=True)
class FullGumbelBatchBenchmarkConfig:
    output_dir: Path
    model_path: Path
    train_shard: Path
    target_result: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    positions: int = 48
    simulations: int = 64
    workers: int = 24
    fixed_batch_size: int = 24
    wait_windows: tuple[float, ...] = (0.00025, 0.0005, 0.001, 0.002)
    repeats: int = 2
    seed: int = 2026082879

    def __post_init__(self) -> None:
        if min(
            self.positions,
            self.simulations,
            self.workers,
            self.fixed_batch_size,
            self.repeats,
        ) <= 0:
            raise ValueError("Full Gumbel benchmark counts must be positive")
        if (
            not self.wait_windows
            or len(set(self.wait_windows)) != len(self.wait_windows)
            or any(wait < 0 for wait in self.wait_windows)
        ):
            raise ValueError("wait-window matrix must be unique and non-negative")


def _network(path: Path) -> HarbiChessNetwork:
    network = HarbiChessNetwork(
        NetworkConfig(
            trunk_channels=16,
            residual_blocks=2,
            policy_channels=4,
            value_channels=2,
            value_hidden=32,
        )
    )
    network.load_weights(str(path))
    return network


def _compare(
    reference: tuple[dict[str, object], ...],
    candidate: tuple[dict[str, object], ...],
) -> dict[str, object]:
    action_mismatches = sum(
        first["selected_action"] != second["selected_action"]
        for first, second in zip(reference, candidate, strict=True)
    )
    visit_mismatches = sum(
        first["root_visits"] != second["root_visits"]
        for first, second in zip(reference, candidate, strict=True)
    )
    maximum_delta = max(
        _determinism_delta(first, second)
        for first, second in zip(reference, candidate, strict=True)
    )
    return {
        "selected_action_mismatches": action_mismatches,
        "root_visit_mismatches": visit_mismatches,
        "maximum_output_delta": maximum_delta,
        "passed": action_mismatches == 0 and visit_mismatches == 0 and maximum_delta <= 1e-12,
    }


def run_full_gumbel_batch_benchmark(config: FullGumbelBatchBenchmarkConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"Full Gumbel benchmark output exists: {config.output_dir}")
    target = json.loads(config.target_result.read_text(encoding="utf-8"))
    identities = tuple(
        str(row["identity"]) for row in target["rows"]["train"][: config.positions]
    )
    if len(identities) != config.positions:
        raise ValueError("target artifact has insufficient benchmark positions")
    rules = PythonChessRules()
    shard = read_shard(config.train_shard, rules=rules)
    records_by_identity = {
        f"{record.game_id}:{record.game_index}:{record.ply}": record
        for record in shard.records
    }
    records = tuple(records_by_identity[identity] for identity in identities)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.EVALUATION,
        mode_detail="AKTAR fixed-batch benchmark · starting",
        run_id=config.output_dir.name,
    )
    store.write_atomic(snapshot)
    arms = []
    reference: tuple[dict[str, object], ...] | None = None
    benchmark_started = time.perf_counter()
    for wait in config.wait_windows:
        batcher = SharedBatchEvaluator(
            MLXPolicyValueBackend(
                _network(config.model_path), fixed_batch_size=config.fixed_batch_size
            ),
            max_batch_size=config.fixed_batch_size,
            max_wait_seconds=wait,
        )
        search = FullGumbelMCTS(
            NeuralPositionEvaluator(batcher, rules=rules),
            rules=rules,
            config=FullGumbelConfig(
                simulations=config.simulations,
                max_considered_actions=16,
                gumbel_scale=0.0,
            ),
        )

        def evaluate(record, current_search=search):
            return _target_row(record, current_search, seed=config.seed, rules=rules)

        with ThreadPoolExecutor(max_workers=min(config.workers, len(records))) as pool:
            tuple(pool.map(evaluate, records[: min(4, len(records))]))
        batcher.reset_statistics()
        repeat_rows = []
        repeat_metrics = []
        try:
            for repeat in range(config.repeats):
                started = time.perf_counter()
                with ThreadPoolExecutor(
                    max_workers=min(config.workers, len(records))
                ) as pool:
                    rows = tuple(pool.map(evaluate, records))
                elapsed = time.perf_counter() - started
                stats = batcher.reset_statistics()
                repeat_rows.append(rows)
                repeat_metrics.append(
                    {
                        "repeat": repeat,
                        "elapsed_seconds": elapsed,
                        "positions_per_second": stats.positions / max(elapsed, 1e-9),
                        "average_batch_size": stats.average_batch_size,
                        "largest_batch": stats.largest_batch,
                        "backend_seconds": stats.backend_seconds,
                        "queue_wait_seconds": stats.queue_wait_seconds,
                    }
                )
        finally:
            batcher.close()
        if reference is None:
            reference = repeat_rows[0]
        equivalence = tuple(_compare(reference, rows) for rows in repeat_rows)
        arms.append(
            {
                "wait_seconds": wait,
                "repeats": repeat_metrics,
                "median_elapsed_seconds": statistics.median(
                    row["elapsed_seconds"] for row in repeat_metrics
                ),
                "equivalence": equivalence,
                "passed": all(row["passed"] for row in equivalence),
            }
        )
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode_detail=f"AKTAR fixed-batch benchmark · wait {wait:g} complete",
        )
        store.write_atomic(snapshot)
    eligible = [arm for arm in arms if arm["passed"]]
    selected = min(eligible, key=lambda arm: arm["median_elapsed_seconds"]) if eligible else None
    config.output_dir.mkdir(parents=True)
    result_path = config.output_dir / "result.json"
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "config": {
                **asdict(config),
                **{
                    name: str(getattr(config, name))
                    for name in (
                        "output_dir",
                        "model_path",
                        "train_shard",
                        "target_result",
                        "telemetry_path",
                    )
                },
            },
            "arms": arms,
            "selected_wait_seconds": None if selected is None else selected["wait_seconds"],
            "passed": selected is not None,
            "elapsed_seconds": time.perf_counter() - benchmark_started,
        },
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            f"AKTAR batch benchmark passed · wait {selected['wait_seconds']:g}"
            if selected is not None
            else "AKTAR batch benchmark failed equivalence"
        ),
    )
    store.write_atomic(snapshot)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-shard", type=Path, required=True)
    parser.add_argument("--target-result", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    result = run_full_gumbel_batch_benchmark(
        FullGumbelBatchBenchmarkConfig(
            output_dir=arguments.output_dir,
            model_path=arguments.model,
            train_shard=arguments.train_shard,
            target_result=arguments.target_result,
            telemetry_path=arguments.telemetry,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
