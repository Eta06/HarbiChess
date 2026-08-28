"""Run the frozen KOPRU function-preserving capacity transfer matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.model_quality import (
    PreparedModelQualityDataset,
    evaluate_model_quality,
    prepare_model_quality_dataset,
)
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.diagnostics import run_tactical_sweep
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.value_oracle import (
    DeterministicTacticalOracle,
    OracleValueEvaluator,
    TacticalOracleConfig,
)
from harbichess.training.batch import GameBalancedSampler, TrainingBatch, build_training_batch
from harbichess.training.learner import LearnerConfig, MLXLearner
from harbichess.training.network_expansion import expand_network_function_preserving


@dataclass(frozen=True, slots=True)
class CapacityVariant:
    name: str
    residual_blocks: int
    policy_channels: int


FROZEN_VARIANTS = (
    CapacityVariant("base", 2, 4),
    CapacityVariant("deep", 4, 4),
    CapacityVariant("head", 2, 8),
    CapacityVariant("deep-head", 4, 8),
)


@dataclass(frozen=True, slots=True)
class CapacityMatrixConfig:
    replay_run_result: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    seed: int = 2026082814
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 2e-4
    tactical_workers: int = 8
    tactical_budget: int = 64
    minimum_top_gain: float = 0.03
    maximum_initial_logit_delta: float = 1e-5
    inference_batches: tuple[int, ...] = (4, 16, 64)
    inference_iterations: int = 20

    def __post_init__(self) -> None:
        if (
            min(
                self.seed,
                self.batch_size,
                self.epochs,
                self.tactical_workers,
                self.tactical_budget,
                self.inference_iterations,
            )
            <= 0
            or self.learning_rate <= 0
            or not self.inference_batches
            or min(self.inference_batches) <= 0
            or not 0 <= self.minimum_top_gain <= 1
            or self.maximum_initial_logit_delta < 0
        ):
            raise ValueError("capacity matrix configuration is invalid")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _network_config(payload: dict[str, object]) -> NetworkConfig:
    return NetworkConfig(
        trunk_channels=int(payload["trunk_channels"]),
        residual_blocks=int(payload["residual_blocks"]),
        policy_channels=int(payload["policy_channels"]),
        value_channels=int(payload["value_channels"]),
        value_hidden=int(payload["value_hidden"]),
    )


def _maximum_logit_delta(
    source: HarbiChessNetwork,
    target: HarbiChessNetwork,
    dataset: PreparedModelQualityDataset,
) -> float:
    maximum = mx.array(0.0)
    for chunk in dataset.chunks:
        source_policy, source_value = source(chunk.inputs)
        target_policy, target_value = target(chunk.inputs)
        maximum = mx.maximum(maximum, mx.max(mx.abs(source_policy - target_policy)))
        maximum = mx.maximum(maximum, mx.max(mx.abs(source_value - target_value)))
    mx.eval(maximum)
    return float(maximum.item())


def _tactical_metrics(network: HarbiChessNetwork, *, budget: int, workers: int, seed: int):
    rules = PythonChessRules()
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=max(8, workers * 2),
    )
    neural = NeuralPositionEvaluator(batcher, rules=rules)
    teacher = OracleValueEvaluator(
        neural,
        DeterministicTacticalOracle(rules=rules, config=TacticalOracleConfig(depth=1)),
    )
    try:
        return run_tactical_sweep(
            teacher,
            rules=rules,
            budgets=(budget,),
            workers=workers,
            seed=seed,
        )
    finally:
        batcher.close()


def _solved(tactical: dict[str, object]) -> tuple[int, int]:
    return int(tactical["raw"]["solved"]), int(tactical["budgets"][0]["solved"])


def _inference_benchmark(
    network: HarbiChessNetwork,
    batch: TrainingBatch,
    *,
    batch_sizes: tuple[int, ...],
    iterations: int,
) -> list[dict[str, float | int]]:
    backend = MLXPolicyValueBackend(network)
    rows = []
    for size in batch_sizes:
        positions = batch.positions[:size]
        actions = tuple(
            tuple(index for index, legal in enumerate(mask) if legal)
            for mask in batch.legal_masks[:size]
        )
        for _ in range(3):
            backend.evaluate_masked(positions, actions)
        started = time.perf_counter()
        for _ in range(iterations):
            backend.evaluate_masked(positions, actions)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "batch_size": size,
                "latency_ms": elapsed / iterations * 1000,
                "positions_per_second": size * iterations / elapsed,
            }
        )
    return rows


def _gate_reasons(
    row: dict[str, object],
    *,
    base_final: dict[str, float],
    baseline_top: float,
    baseline_tactical: tuple[int, int],
    config: CapacityMatrixConfig,
) -> tuple[str, ...]:
    reasons = []
    final = row["checkpoints"][-1]["quality"]
    if row["initial_logit_delta"] > config.maximum_initial_logit_delta:
        reasons.append("function-preserving initialization exceeded tolerance")
    if final["teacher_policy_cross_entropy"] > base_final["teacher_policy_cross_entropy"]:
        reasons.append("validation policy cross-entropy did not beat base control")
    if final["teacher_top_action_agreement"] <= baseline_top:
        reasons.append("teacher top-action agreement did not beat release baseline")
    if final["teacher_top_action_agreement"] < (
        base_final["teacher_top_action_agreement"] + config.minimum_top_gain
    ):
        reasons.append("teacher top-action gain over base control was below 3pp")
    if any(
        candidate < baseline
        for candidate, baseline in zip(
            _solved(row["tactical"]), baseline_tactical, strict=True
        )
    ):
        reasons.append("tactical solve count regressed")
    if not row["gradients_finite"]:
        reasons.append("loss or gradients became non-finite")
    return tuple(reasons)


def run_capacity_matrix(config: CapacityMatrixConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"capacity matrix output already exists: {config.output_dir}")
    replay_result = json.loads(config.replay_run_result.read_text(encoding="utf-8"))
    if replay_result.get("mode") != "generation_only" or not replay_result.get("passed"):
        raise ValueError("capacity matrix requires the qualified generation-only replay")
    train_path = config.replay_run_result.parent / "replay" / "train-00000.jsonl.gz"
    validation_path = config.replay_run_result.parent / "replay" / "validation-00000.jsonl.gz"

    timings: dict[str, float] = {}
    started = time.perf_counter()
    train = read_shard(train_path).records
    validation = read_shard(validation_path).records
    timings["replay_read_seconds"] = time.perf_counter() - started
    if {record.game_id for record in train} & {record.game_id for record in validation}:
        raise ValueError("capacity matrix detected train/validation game leakage")

    started = time.perf_counter()
    training_batch = build_training_batch(train)
    timings["training_batch_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    prepared_training = MLXLearner.prepare_batch(training_batch)
    validation_dataset = prepare_model_quality_dataset(validation)
    timings["mlx_preparation_seconds"] = time.perf_counter() - started

    source_config = _network_config(replay_result["config"])
    baseline_path = Path(replay_result["baseline"]["path"])
    source = HarbiChessNetwork(source_config)
    source.load_weights(str(baseline_path))
    baseline_quality = evaluate_model_quality(source, validation_dataset)
    tactical_baseline = HarbiChessNetwork(source_config)
    tactical_baseline.load_weights(str(baseline_path))
    baseline_tactical_payload = _tactical_metrics(
        tactical_baseline,
        budget=config.tactical_budget,
        workers=config.tactical_workers,
        seed=config.seed,
    )
    baseline_tactical = _solved(baseline_tactical_payload)
    steps_per_epoch = math.ceil(len(train) / config.batch_size)
    checkpoint_steps = (
        0,
        steps_per_epoch // 2,
        steps_per_epoch,
        steps_per_epoch * config.epochs,
    )

    store = SnapshotStore(config.telemetry_path)
    dashboard = store.read()
    rows = []
    for variant_index, variant in enumerate(FROZEN_VARIANTS):
        dashboard = replace(
            dashboard,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.TRAINING,
            mode_detail=f"KOPRU capacity matrix · {variant.name} ({variant_index + 1}/4)",
            pilot_status=PilotStatus.TRAINING,
        )
        store.write_atomic(dashboard)
        target_config = replace(
            source_config,
            residual_blocks=variant.residual_blocks,
            policy_channels=variant.policy_channels,
        )
        network = expand_network_function_preserving(source, target_config)
        initial_delta = _maximum_logit_delta(source, network, validation_dataset)
        learner = MLXLearner(
            network,
            config=LearnerConfig(
                learning_rate=config.learning_rate,
                weight_decay=0.0,
                policy_weight=1.0,
                value_weight=0.0,
            ),
        )
        sampler = GameBalancedSampler(train, seed=config.seed)
        checkpoints = []
        gradients_finite = True
        maximum_gradient_norm = 0.0
        train_started = time.perf_counter()
        for step in range(checkpoint_steps[-1] + 1):
            if step in checkpoint_steps:
                checkpoints.append(
                    {
                        "step": step,
                        "epochs": step / steps_per_epoch,
                        "quality": evaluate_model_quality(network, validation_dataset).to_dict(),
                    }
                )
            if step == checkpoint_steps[-1]:
                break
            metrics = learner.train_step(
                prepared_training.select(sampler.sample_indices(config.batch_size))
            )
            gradients_finite &= all(
                math.isfinite(value)
                for value in (
                    metrics.policy_loss,
                    metrics.value_loss,
                    metrics.total_loss,
                    metrics.unclipped_gradient_norm,
                )
            )
            maximum_gradient_norm = max(maximum_gradient_norm, metrics.gradient_norm)
        training_seconds = time.perf_counter() - train_started
        tactical = _tactical_metrics(
            network,
            budget=config.tactical_budget,
            workers=config.tactical_workers,
            seed=config.seed,
        )
        inference = _inference_benchmark(
            network,
            training_batch,
            batch_sizes=config.inference_batches,
            iterations=config.inference_iterations,
        )
        rows.append(
            {
                "variant": asdict(variant),
                "network_config": asdict(target_config),
                "parameters": network.parameter_count,
                "initial_logit_delta": initial_delta,
                "training_seconds": training_seconds,
                "training_positions_per_second": (
                    checkpoint_steps[-1] * config.batch_size / training_seconds
                ),
                "maximum_gradient_norm": maximum_gradient_norm,
                "gradients_finite": gradients_finite,
                "checkpoints": checkpoints,
                "tactical": tactical,
                "inference": inference,
            }
        )

    base_final = rows[0]["checkpoints"][-1]["quality"]
    passing = []
    for row in rows:
        reasons = _gate_reasons(
            row,
            base_final=base_final,
            baseline_top=baseline_quality.teacher_top_action_agreement,
            baseline_tactical=baseline_tactical,
            config=config,
        )
        row["passed"] = not reasons
        row["gate_reasons"] = reasons
        if not reasons:
            passing.append(row["variant"]["name"])

    config.output_dir.mkdir(parents=True)
    result_path = config.output_dir / "result.json"
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": _source_commit(),
        "config": {
            **asdict(config),
            "replay_run_result": str(config.replay_run_result),
            "output_dir": str(config.output_dir),
            "telemetry_path": str(config.telemetry_path),
        },
        "replay": {
            "train_samples": len(train),
            "validation_samples": len(validation),
            "steps_per_epoch": steps_per_epoch,
            "checkpoint_sha256": _sha256(baseline_path),
        },
        "baseline": {
            "quality": baseline_quality.to_dict(),
            "tactical": baseline_tactical_payload,
        },
        "timings": timings,
        "variants": rows,
        "passing_variants": passing,
        "passed": bool(passing),
        "arena_authorized": False,
        "generation_authorized": False,
    }
    _atomic_json(result_path, payload)
    dashboard = replace(
        dashboard,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            f"KOPRU capacity matrix passed · {', '.join(passing)} require confirmation"
            if passing
            else "KOPRU capacity matrix failed · learner and arena remain blocked"
        ),
        pilot_status=PilotStatus.PASSED if passing else PilotStatus.FAILED,
        promotion_ready=False,
    )
    store.write_atomic(dashboard)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-run-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    path = run_capacity_matrix(
        CapacityMatrixConfig(
            replay_run_result=arguments.replay_run_result,
            output_dir=arguments.output_dir,
            telemetry_path=arguments.telemetry,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
