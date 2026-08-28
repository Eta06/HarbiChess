"""Matched-control auxiliary short-horizon value transfer experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import mean, pstdev

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import RunMode, SnapshotStore
from harbichess.evaluation.model_quality import (
    evaluate_model_quality,
    prepare_model_quality_dataset,
)
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.diagnostics import run_tactical_sweep
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.learner import MLXLearner, PreparedTrainingBatch


@dataclass(frozen=True, slots=True)
class ShortHorizonConfig:
    output_dir: Path
    replay_dir: Path
    model_path: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    model_sha256: str = "5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca"
    trunk_channels: int = 16
    residual_blocks: int = 2
    policy_channels: int = 4
    value_channels: int = 2
    value_hidden: int = 32
    steps: int = 474
    batch_size: int = 64
    learning_rate: float = 2e-4
    auxiliary_lambda: float = 0.8
    auxiliary_weight: float = 0.25
    validation_interval: int = 79
    seed: int = 2026082869
    workers: int = 8

    def __post_init__(self) -> None:
        if (
            self.steps <= 0
            or self.batch_size <= 0
            or self.learning_rate <= 0
            or not 0.0 <= self.auxiliary_lambda < 1.0
            or self.auxiliary_weight <= 0
            or self.validation_interval <= 0
            or self.steps % self.validation_interval
            or self.seed < 0
            or self.workers <= 0
        ):
            raise ValueError("short-horizon configuration is invalid")


class AuxiliaryTrainingNetwork(nn.Module):
    def __init__(self, network: HarbiChessNetwork) -> None:
        super().__init__()
        self.network = network
        self.short_value_output = nn.Linear(network.config.value_hidden, 1)

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        trunk = self.network._trunk(inputs)
        policy = self.network.policy_linear(self.network._policy_features(trunk))
        value = nn.relu(self.network.value_conv(trunk)).reshape(trunk.shape[0], -1)
        value_hidden = nn.relu(self.network.value_hidden(value))
        wdl = self.network.value_output(value_hidden)
        short = mx.tanh(self.short_value_output(value_hidden)).squeeze(-1)
        return policy, wdl, short


def short_horizon_targets(
    records: tuple[ReplayRecord, ...], *, coefficient: float
) -> tuple[float, ...]:
    """Build perspective-correct exponential future-search targets by game."""

    if not records or not 0.0 <= coefficient < 1.0:
        raise ValueError("short-horizon targets require records and lambda in [0, 1)")
    by_game: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_game.setdefault(record.game_id, []).append(index)
    targets = [0.0] * len(records)
    for indices in by_game.values():
        ordered = sorted(indices, key=lambda index: records[index].ply)
        if any(
            records[current].ply + 1 != records[following].ply
            for current, following in pairwise(ordered)
        ):
            raise ValueError("short-horizon records must contain consecutive game plies")
        final = records[ordered[-1]]
        future = (
            float(final.outcome_value)
            if final.outcome_value is not None
            else float(final.root_value)
        )
        for index in reversed(ordered):
            record = records[index]
            target = (1.0 - coefficient) * float(record.root_value) + coefficient * future
            targets[index] = max(-1.0, min(1.0, target))
            future = -targets[index]
    return tuple(targets)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
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


def _network_config(config: ShortHorizonConfig) -> NetworkConfig:
    return NetworkConfig(
        trunk_channels=config.trunk_channels,
        residual_blocks=config.residual_blocks,
        policy_channels=config.policy_channels,
        value_channels=config.value_channels,
        value_hidden=config.value_hidden,
    )


def _load_records(replay_dir: Path, split: str) -> tuple[ReplayRecord, ...]:
    paths = tuple(sorted(replay_dir.glob(f"{split}-*.jsonl.gz")))
    if not paths:
        raise ValueError(f"no {split} replay shards found")
    return tuple(record for path in paths for record in read_shard(path).records)


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean, right_mean = mean(left), mean(right)
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return (
        sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
        / denominator
        if denominator
        else 0.0
    )


def value_distribution(
    network: HarbiChessNetwork,
    prepared,
) -> dict[str, float | int]:
    predictions: list[float] = []
    outcomes: list[float] = []
    for chunk in prepared.chunks:
        _, wdl_logits = network(chunk.inputs)
        probabilities = mx.softmax(wdl_logits, axis=1)
        mx.eval(probabilities)
        for record, row in zip(chunk.records, probabilities.tolist(), strict=True):
            if record.outcome_value is None:
                continue
            predictions.append(float(row[0] - row[2]))
            outcomes.append(float(record.outcome_value))
    return {
        "samples": len(predictions),
        "mean": mean(predictions),
        "standard_deviation": pstdev(predictions),
        "outcome_pearson": _pearson(predictions, outcomes),
        "minimum": min(predictions),
        "maximum": max(predictions),
    }


class AuxiliaryLearner:
    def __init__(
        self,
        network: AuxiliaryTrainingNetwork,
        *,
        learning_rate: float,
        auxiliary_weight: float,
    ) -> None:
        self.network = network
        self.auxiliary_weight = auxiliary_weight
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
        self.loss_and_grad = nn.value_and_grad(network, self._loss)

    def _loss(
        self,
        inputs: mx.array,
        policies: mx.array,
        legal_masks: mx.array,
        wdl: mx.array,
        value_weights: mx.array,
        auxiliary: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        policy_logits, wdl_logits, short = self.network(inputs)
        policy_logits = mx.where(legal_masks, policy_logits, mx.array(-1e9))
        policy_loss = nn.losses.cross_entropy(policy_logits, policies, reduction="mean")
        value_rows = nn.losses.cross_entropy(wdl_logits, wdl, reduction="none")
        value_loss = mx.sum(value_rows * value_weights) / mx.maximum(
            mx.sum(value_weights), mx.array(1.0)
        )
        errors = mx.abs(short - auxiliary)
        auxiliary_loss = mx.mean(mx.where(errors <= 1.0, 0.5 * errors**2, errors - 0.5))
        total = policy_loss + value_loss + self.auxiliary_weight * auxiliary_loss
        return total, policy_loss, value_loss, auxiliary_loss

    def train_step(
        self,
        batch: PreparedTrainingBatch,
        auxiliary: mx.array,
    ) -> tuple[float, float, float, float, float]:
        (total, policy, value, short), gradients = self.loss_and_grad(
            batch.inputs,
            batch.policy_targets,
            batch.legal_masks,
            batch.wdl_targets,
            batch.value_weights,
            auxiliary,
        )
        gradients, norm = optim.clip_grad_norm(gradients, 5.0)
        mx.eval(total, policy, value, short, norm, gradients)
        values = (total, policy, value, short, norm)
        if not all(math.isfinite(float(item.item())) for item in values):
            raise ValueError("non-finite short-horizon training metric")
        self.optimizer.update(self.network, gradients)
        mx.eval(self.network.parameters(), self.optimizer.state)
        return tuple(float(item.item()) for item in values)  # type: ignore[return-value]

    def evaluate(
        self, batch: PreparedTrainingBatch, auxiliary: mx.array
    ) -> tuple[float, float, float, float]:
        values = self._loss(
            batch.inputs,
            batch.policy_targets,
            batch.legal_masks,
            batch.wdl_targets,
            batch.value_weights,
            auxiliary,
        )
        mx.eval(*values)
        return tuple(float(value.item()) for value in values)  # type: ignore[return-value]


def _tactical(network: HarbiChessNetwork, config: ShortHorizonConfig) -> dict[str, object]:
    rules = PythonChessRules()
    with SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=config.workers,
        max_wait_seconds=0.00025,
    ) as batcher:
        return run_tactical_sweep(
            NeuralPositionEvaluator(batcher, rules=rules),
            rules=rules,
            budgets=(128, 256),
            workers=config.workers,
            seed=config.seed,
        )


def evaluate_auxiliary_gate(
    baseline: dict[str, object],
    control: dict[str, object],
    auxiliary: dict[str, object],
) -> tuple[bool, tuple[str, ...]]:
    reasons = []
    baseline_quality = baseline["quality"]
    control_quality = control["quality"]
    auxiliary_quality = auxiliary["quality"]
    baseline_value = baseline["value_distribution"]
    control_value = control["value_distribution"]
    auxiliary_value = auxiliary["value_distribution"]
    assert isinstance(baseline_quality, dict)
    assert isinstance(control_quality, dict)
    assert isinstance(auxiliary_quality, dict)
    assert isinstance(baseline_value, dict)
    assert isinstance(control_value, dict)
    assert isinstance(auxiliary_value, dict)
    value_ce = float(auxiliary_quality["value_cross_entropy"])
    if value_ce > float(baseline_quality["value_cross_entropy"]) * 1.01:
        reasons.append("auxiliary WDL loss regressed versus baseline")
    if value_ce > float(control_quality["value_cross_entropy"]) * 1.01:
        reasons.append("auxiliary WDL loss regressed versus control")
    correlation = float(auxiliary_value["outcome_pearson"])
    if correlation < float(baseline_value["outcome_pearson"]) + 0.05:
        reasons.append("auxiliary outcome correlation gain was below 0.05")
    if correlation < float(control_value["outcome_pearson"]):
        reasons.append("auxiliary outcome correlation was below control")
    if float(auxiliary_value["standard_deviation"]) < 0.02:
        reasons.append("auxiliary value predictions remained collapsed")
    if float(auxiliary_quality["teacher_policy_cross_entropy"]) > (
        float(control_quality["teacher_policy_cross_entropy"]) * 1.01
    ):
        reasons.append("auxiliary policy loss regressed versus control")
    base_tactical = baseline["tactical"]
    aux_tactical = auxiliary["tactical"]
    assert isinstance(base_tactical, dict) and isinstance(aux_tactical, dict)
    if int(aux_tactical["raw"]["solved"]) < int(base_tactical["raw"]["solved"]):  # type: ignore[index]
        reasons.append("auxiliary raw tactical solve count regressed")
    base_budgets = {row["budget"]: row for row in base_tactical["budgets"]}  # type: ignore[index]
    aux_budgets = {row["budget"]: row for row in aux_tactical["budgets"]}  # type: ignore[index]
    for budget in (128, 256):
        if int(aux_budgets[budget]["solved"]) < int(base_budgets[budget]["solved"]):
            reasons.append(f"auxiliary {budget} tactical solve count regressed")
    base_solved = {
        row["case"] for row in base_budgets[128]["cases"] if row["solved"]
    }
    aux_solved = {row["case"] for row in aux_budgets[256]["cases"] if row["solved"]}
    if base_solved - aux_solved:
        reasons.append("auxiliary 256 lost a baseline-128 tactical case")
    if float(auxiliary["maximum_gradient_norm"]) > 5.0:
        reasons.append("auxiliary gradient norm exceeded 5.0")
    return not reasons, tuple(reasons)


def run_short_horizon_value(config: ShortHorizonConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"short-horizon output exists: {config.output_dir}")
    if _sha256(config.model_path) != config.model_sha256:
        raise ValueError("short-horizon baseline checksum mismatch")
    train = _load_records(config.replay_dir, "train")
    validation = _load_records(config.replay_dir, "validation")
    if {record.game_id for record in train} & {record.game_id for record in validation}:
        raise ValueError("short-horizon train/validation leakage")
    rules = PythonChessRules()
    train_batch = MLXLearner.prepare_batch(build_training_batch(train, rules=rules))
    validation_batch = MLXLearner.prepare_batch(build_training_batch(validation, rules=rules))
    train_aux = mx.array(
        short_horizon_targets(train, coefficient=config.auxiliary_lambda), dtype=mx.float32
    )
    validation_aux = mx.array(
        short_horizon_targets(validation, coefficient=config.auxiliary_lambda),
        dtype=mx.float32,
    )
    mx.eval(train_aux, validation_aux)
    quality_data = prepare_model_quality_dataset(validation, rules=rules)
    network_config = _network_config(config)

    baseline_network = HarbiChessNetwork(network_config)
    baseline_network.load_weights(str(config.model_path))
    baseline = {
        "quality": evaluate_model_quality(baseline_network, quality_data).to_dict(),
        "value_distribution": value_distribution(baseline_network, quality_data),
        "tactical": _tactical(baseline_network, config),
    }
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="ESAS short-horizon value · matched control",
        run_id="esas-short-horizon-value-20260828-01",
        pilot_steps_planned=config.steps * 2,
        pilot_steps_completed=0,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    arms: dict[str, dict[str, object]] = {}
    for arm_index, (name, weight) in enumerate(
        (("control", 0.0), ("auxiliary", config.auxiliary_weight))
    ):
        mx.random.seed(config.seed)
        base = HarbiChessNetwork(network_config)
        base.load_weights(str(config.model_path))
        learner = AuxiliaryLearner(
            AuxiliaryTrainingNetwork(base),
            learning_rate=config.learning_rate,
            auxiliary_weight=weight,
        )
        sampler = GameBalancedSampler(train, seed=config.seed)
        curve = []
        maximum_gradient = 0.0
        arm_started = time.perf_counter()
        for step in range(1, config.steps + 1):
            indices = sampler.sample_indices(config.batch_size)
            selected = train_batch.select(indices)
            metrics = learner.train_step(selected, mx.take(train_aux, mx.array(indices), axis=0))
            maximum_gradient = max(maximum_gradient, metrics[-1])
            if step % config.validation_interval == 0:
                validation_metrics = learner.evaluate(validation_batch, validation_aux)
                curve.append({"step": step, "losses": validation_metrics})
                snapshot = replace(
                    snapshot,
                    updated_at=datetime.now(UTC).isoformat(),
                    mode_detail=f"ESAS short-horizon value · {name} · {step}/{config.steps}",
                    training_step=arm_index * config.steps + step,
                    pilot_steps_completed=arm_index * config.steps + step,
                    policy_loss=metrics[1],
                    value_loss=metrics[2],
                    total_loss=metrics[0],
                    training_elapsed_seconds=time.perf_counter() - started,
                    session_elapsed_seconds=time.perf_counter() - started,
                    positions_per_second=(arm_index * config.steps + step)
                    * config.batch_size
                    / max(time.perf_counter() - started, 1e-9),
                )
                store.write_atomic(snapshot)
        model_path = config.output_dir / name / "model.safetensors"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        base.save_weights(str(model_path))
        arms[name] = {
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "quality": evaluate_model_quality(base, quality_data).to_dict(),
            "value_distribution": value_distribution(base, quality_data),
            "tactical": _tactical(base, config),
            "maximum_gradient_norm": maximum_gradient,
            "curve": curve,
            "elapsed_seconds": time.perf_counter() - arm_started,
        }
    passed, reasons = evaluate_auxiliary_gate(baseline, arms["control"], arms["auxiliary"])
    result_path = config.output_dir / "result.json"
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "passed": passed,
            "reasons": reasons,
            "system_teacher_rerun_authorized": passed,
            "learner_latest_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
            "config": {
                **asdict(config),
                "output_dir": str(config.output_dir),
                "replay_dir": str(config.replay_dir),
                "model_path": str(config.model_path),
                "telemetry_path": str(config.telemetry_path),
            },
            "data": {
                "train": len(train),
                "validation": len(validation),
                "train_known_wdl": sum(row.outcome_value is not None for row in train),
                "validation_known_wdl": sum(
                    row.outcome_value is not None for row in validation
                ),
            },
            "baseline": baseline,
            "arms": arms,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "ESAS auxiliary value passed · system teacher rerun authorized"
            if passed
            else "ESAS auxiliary value failed · learner remains blocked"
        ),
    )
    store.write_atomic(snapshot)
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_short_horizon_value(
        ShortHorizonConfig(
            output_dir=arguments.output_dir,
            replay_dir=arguments.replay_dir,
            model_path=arguments.model,
            telemetry_path=arguments.telemetry,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
