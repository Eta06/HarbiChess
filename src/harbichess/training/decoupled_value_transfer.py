"""Qualify decoupled auxiliary material and production WDL heads."""

from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.corrected_replay_value_transfer import (
    CorrectedReplayValueTransferConfig,
    _load_games,
    _split_games,
)
from harbichess.evaluation.deterministic_value_probe import _prepare, _round_robin
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.training.batch import GameBalancedSampler
from harbichess.training.full_gumbel_transfer import _network, _snapshot
from harbichess.training.invariant_wdl_transfer import _wdl_quality
from harbichess.training.joint_policy_value_transfer import (
    OutcomeGameBalancedSampler,
    _parameter_hash,
    _sha256,
    _value_gate_reasons,
)


@dataclass(frozen=True, slots=True)
class DecoupledValueTransferConfig:
    output_dir: Path
    model_path: Path
    runs_root: Path = Path("artifacts/runs")
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    material_train_positions: int = 8192
    material_validation_positions: int = 4096
    material_steps: int = 200
    material_learning_rate: float = 2e-3
    wdl_steps: int = 400
    wdl_learning_rate: float = 5e-4
    batch_size: int = 64
    validation_interval: int = 20
    material_seed: int = 2026083061
    wdl_seed: int = 2026083073
    wdl_sampling_mode: str = "outcome"

    def __post_init__(self) -> None:
        if (
            min(
                self.material_train_positions,
                self.material_validation_positions,
                self.material_steps,
                self.material_learning_rate,
                self.wdl_steps,
                self.wdl_learning_rate,
                self.batch_size,
                self.validation_interval,
                self.material_seed,
                self.wdl_seed,
            )
            <= 0
            or self.material_steps % self.validation_interval
            or self.wdl_steps % self.validation_interval
        ):
            raise ValueError("decoupled value transfer configuration is invalid")
        if self.wdl_sampling_mode not in {"outcome", "mixed", "natural"}:
            raise ValueError("WDL sampling mode must be outcome, mixed, or natural")


_MATERIAL_PREFIX = ("material_value_linear.",)
_PRODUCTION_PREFIXES = (
    "invariant_value_linear.",
    "value_tower_stem.",
    "value_tower_blocks.",
    "value_tower_hidden.",
    "value_tower_output.",
)
_NEW_PREFIXES = (*_MATERIAL_PREFIX, *_PRODUCTION_PREFIXES)


def _material_quality(network, inputs: mx.array, targets: mx.array) -> dict[str, float | int]:
    predictions = network.material_value(inputs)
    mx.eval(predictions)
    predicted = predictions.tolist()
    expected = targets.tolist()
    errors = [a - b for a, b in zip(predicted, expected, strict=True)]
    left_mean = sum(predicted) / len(predicted)
    right_mean = sum(expected) / len(expected)
    left = [value - left_mean for value in predicted]
    right = [value - right_mean for value in expected]
    denominator = math.sqrt(
        sum(value * value for value in left) * sum(value * value for value in right)
    )
    pearson = (
        sum(a * b for a, b in zip(left, right, strict=True)) / denominator if denominator else 0.0
    )
    return {
        "positions": len(expected),
        "mse": sum(error * error for error in errors) / len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "pearson": pearson,
    }


def _material_reasons(
    baseline: dict[str, float | int], candidate: dict[str, float | int]
) -> tuple[str, ...]:
    reasons = []
    if float(candidate["mse"]) > float(baseline["mse"]) * 0.5:
        reasons.append("auxiliary material MSE did not improve by 50 percent")
    if float(candidate["pearson"]) < 0.80:
        reasons.append("auxiliary material Pearson is below 0.80")
    if float(candidate["mae"]) > 0.05:
        reasons.append("auxiliary material MAE exceeds 0.05")
    return tuple(reasons)


class _MaterialLearner:
    def __init__(self, network, *, learning_rate: float) -> None:
        self.network = network
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.loss_and_grad = nn.value_and_grad(network, self._loss)

    @staticmethod
    def _loss(network, inputs: mx.array, targets: mx.array) -> mx.array:
        return mx.mean(mx.square(network.material_value(inputs) - targets))

    def step(self, inputs: mx.array, targets: mx.array) -> float:
        loss, gradients = self.loss_and_grad(self.network, inputs, targets)
        self.optimizer.update(self.network, gradients)
        mx.eval(loss, self.network.parameters(), self.optimizer.state)
        return float(loss.item())


class _WDLLearner:
    def __init__(self, network, *, learning_rate: float) -> None:
        self.network = network
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.loss_and_grad = nn.value_and_grad(network, self._loss)

    @staticmethod
    def _loss(network, inputs: mx.array, targets: mx.array) -> mx.array:
        return nn.losses.cross_entropy(network(inputs)[1], targets, reduction="mean")

    def step(self, inputs: mx.array, targets: mx.array) -> tuple[float, float]:
        loss, gradients = self.loss_and_grad(self.network, inputs, targets)
        gradients, norm = optim.clip_grad_norm(gradients, 5.0)
        mx.eval(loss, norm, gradients)
        if not math.isfinite(float(loss.item())) or not math.isfinite(float(norm.item())):
            raise RuntimeError("decoupled WDL transfer produced non-finite gradients")
        self.optimizer.update(self.network, gradients)
        mx.eval(self.network.parameters(), self.optimizer.state)
        return float(loss.item()), float(norm.item())


class _MixedWDLSampler:
    """Combine fixed-size outcome-balanced and natural game-balanced draws."""

    def __init__(self, records, *, seed: int) -> None:
        self._outcome = OutcomeGameBalancedSampler(records, seed=seed)
        self._natural = GameBalancedSampler(records, seed=seed + 1)
        self._rng = random.Random(seed + 2)

    def sample_indices(self, batch_size: int) -> tuple[int, ...]:
        if batch_size <= 1:
            raise ValueError("mixed WDL sampler requires at least two rows")
        outcome_size = batch_size // 2
        selected = [
            *self._outcome.sample_indices(outcome_size),
            *self._natural.sample_indices(batch_size - outcome_size),
        ]
        self._rng.shuffle(selected)
        return tuple(selected)


def _wdl_sampler(records, *, mode: str, seed: int):
    if mode == "outcome":
        return OutcomeGameBalancedSampler(records, seed=seed)
    if mode == "natural":
        return GameBalancedSampler(records, seed=seed)
    if mode == "mixed":
        return _MixedWDLSampler(records, seed=seed)
    raise ValueError(f"unsupported WDL sampling mode: {mode}")


def _train_material(
    base,
    train: tuple[mx.array, mx.array],
    validation: tuple[mx.array, mx.array],
    *,
    config: DecoupledValueTransferConfig,
    store: SnapshotStore,
    snapshot,
) -> tuple[dict[str, object], object, HarbiChessDecoupledValueNetwork]:
    network = HarbiChessDecoupledValueNetwork.from_base(base)
    network.freeze_to_material_head()
    frozen_before = _parameter_hash(network, excluded_prefixes=_MATERIAL_PREFIX)
    baseline = _material_quality(network, *validation)
    learner = _MaterialLearner(network, learning_rate=config.material_learning_rate)
    rng = random.Random(config.material_seed)
    best = (float(baseline["mse"]), 0, _snapshot(network))
    curve = []
    for step in range(1, config.material_steps + 1):
        indices = tuple(rng.randrange(train[0].shape[0]) for _ in range(config.batch_size))
        rows = mx.array(indices, dtype=mx.int32)
        loss = learner.step(mx.take(train[0], rows, axis=0), mx.take(train[1], rows, axis=0))
        if step % config.validation_interval:
            continue
        quality = _material_quality(network, *validation)
        curve.append({"step": step, "batch_loss": loss, "validation": quality})
        if float(quality["mse"]) < best[0]:
            best = (float(quality["mse"]), step, _snapshot(network))
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode_detail=f"MIHVER decoupled material · {step}/{config.material_steps}",
            pilot_steps_completed=step,
            training_step=step,
            value_loss=float(quality["mse"]),
        )
        store.write_atomic(snapshot)
    network.load_weights(list(best[2]))
    selected = _material_quality(network, *validation)
    reasons = list(_material_reasons(baseline, selected))
    frozen_after = _parameter_hash(network, excluded_prefixes=_MATERIAL_PREFIX)
    if frozen_before != frozen_after:
        reasons.append("material training changed a frozen production parameter")
    return (
        {
            "baseline": baseline,
            "selected_step": best[1],
            "selected": selected,
            "passed": not reasons,
            "reasons": reasons,
            "frozen_hash_before": frozen_before,
            "frozen_hash_after": frozen_after,
            "curve": curve,
        },
        snapshot,
        network,
    )


def _train_wdl_arm(
    label: str,
    material_network,
    train_records,
    validation_records,
    train: tuple[mx.array, mx.array],
    validation: tuple[mx.array, mx.array],
    release_wdl: dict[str, object],
    *,
    config: DecoupledValueTransferConfig,
    global_only: bool,
    store: SnapshotStore,
    snapshot,
    arm_index: int,
) -> tuple[dict[str, object], object]:
    network = HarbiChessDecoupledValueNetwork.from_base(_network())
    network.load_weights(list(_snapshot(material_network)))
    if global_only:
        network.freeze_to_global_wdl()
    else:
        network.freeze_to_global_tower_wdl()
    frozen_before = _parameter_hash(network, excluded_prefixes=_PRODUCTION_PREFIXES)
    material_before = _material_quality(network, *validation)
    train_outcomes = tuple(int(record.outcome_value) for record in train_records)
    validation_outcomes = tuple(int(record.outcome_value) for record in validation_records)
    labels = mx.array([{1: 0, 0: 1, -1: 2}[value] for value in train_outcomes], dtype=mx.int32)
    sampler = _wdl_sampler(
        train_records,
        mode=config.wdl_sampling_mode,
        seed=config.wdl_seed,
    )
    learner = _WDLLearner(network, learning_rate=config.wdl_learning_rate)
    best = (math.inf, 0, _snapshot(network))
    eligible = []
    curve = []
    maximum_gradient_norm = 0.0
    for step in range(1, config.wdl_steps + 1):
        indices = sampler.sample_indices(config.batch_size)
        rows = mx.array(indices, dtype=mx.int32)
        loss, norm = learner.step(mx.take(train[0], rows, axis=0), mx.take(labels, rows, axis=0))
        maximum_gradient_norm = max(maximum_gradient_norm, norm)
        if step % config.validation_interval:
            continue
        validation_wdl = _wdl_quality(network, validation[0], validation_outcomes)
        train_wdl = _wdl_quality(network, train[0], train_outcomes)
        reasons = _value_gate_reasons(release_wdl, validation_wdl)
        macro = float(validation_wdl["macro_cross_entropy"])
        weights = _snapshot(network)
        if macro < best[0]:
            best = (macro, step, weights)
        if not reasons:
            eligible.append((macro, step, weights))
        curve.append(
            {
                "step": step,
                "batch_loss": loss,
                "train_wdl": train_wdl,
                "validation_wdl": validation_wdl,
                "gate_reasons": reasons,
            }
        )
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode_detail=f"MIHVER decoupled WDL · {label} · {step}/{config.wdl_steps}",
            pilot_steps_completed=config.material_steps + arm_index * config.wdl_steps + step,
            training_step=config.material_steps + arm_index * config.wdl_steps + step,
            value_loss=float(validation_wdl["cross_entropy"]),
        )
        store.write_atomic(snapshot)
    selected_macro, selected_step, weights = (
        min(eligible, key=lambda row: (row[0], row[1])) if eligible else best
    )
    network.load_weights(list(weights))
    selected_wdl = _wdl_quality(network, validation[0], validation_outcomes)
    selected_material = _material_quality(network, *validation)
    reasons = list(_value_gate_reasons(release_wdl, selected_wdl))
    frozen_after = _parameter_hash(network, excluded_prefixes=_PRODUCTION_PREFIXES)
    if frozen_before != frozen_after:
        reasons.append("WDL training changed a frozen release or material parameter")
    if selected_material != material_before:
        reasons.append("WDL training changed auxiliary material predictions")
    model_path = config.output_dir / "arms" / label / "model.safetensors"
    model_path.parent.mkdir(parents=True)
    temporary = model_path.with_name(".model.tmp.safetensors")
    network.save_weights(str(temporary))
    os.replace(temporary, model_path)
    return {
        "label": label,
        "selected_step": selected_step,
        "selected_macro_wdl_ce": selected_macro,
        "selected_wdl": selected_wdl,
        "material_before": material_before,
        "material_after": selected_material,
        "passed": bool(eligible) and not reasons,
        "reasons": reasons,
        "eligible_checkpoints": [step for _, step, _ in eligible],
        "maximum_gradient_norm": maximum_gradient_norm,
        "frozen_hash_before": frozen_before,
        "frozen_hash_after": frozen_after,
        "curve": curve,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
    }, snapshot


def _select_wdl_arm(arms: dict[str, dict[str, object]]) -> str | None:
    passed = [label for label, arm in arms.items() if arm["passed"]]
    if not passed:
        return None
    return min(
        passed,
        key=lambda label: (
            float(arms[label]["selected_macro_wdl_ce"]),
            label == "global-tower-wdl",
        ),
    )


def run_decoupled_value_transfer(config: DecoupledValueTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"decoupled value output exists: {config.output_dir}")
    pool_config = CorrectedReplayValueTransferConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        runs_root=config.runs_root,
    )
    games, provenance = _load_games(pool_config)
    train_records, validation_records, split = _split_games(games, seed=pool_config.seed)
    material_train_records = _round_robin(train_records, config.material_train_positions)
    material_validation_records = _round_robin(
        validation_records, config.material_validation_positions
    )
    rules = PythonChessRules()
    material_train = _prepare(material_train_records, rules)
    material_validation = _prepare(material_validation_records, rules)
    wdl_train = _prepare(train_records, rules)
    wdl_validation = _prepare(validation_records, rules)
    base = _network()
    base.load_weights(str(config.model_path))
    release_wdl = _wdl_quality(
        base,
        wdl_validation[0],
        tuple(int(record.outcome_value) for record in validation_records),
    )
    config.output_dir.mkdir(parents=True)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="MIHVER decoupled value qualification",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.material_steps + 2 * config.wdl_steps,
        pilot_steps_completed=0,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    material, snapshot, material_network = _train_material(
        base,
        material_train,
        material_validation,
        config=config,
        store=store,
        snapshot=snapshot,
    )
    material_model_path = config.output_dir / "material" / "model.safetensors"
    material_model_path.parent.mkdir()
    material_network.save_weights(str(material_model_path))
    wdl_arms = {}
    if material["passed"]:
        for index, (label, global_only) in enumerate(
            (("global-wdl", True), ("global-tower-wdl", False))
        ):
            wdl_arms[label], snapshot = _train_wdl_arm(
                label,
                material_network,
                train_records,
                validation_records,
                wdl_train,
                wdl_validation,
                release_wdl,
                config=config,
                global_only=global_only,
                store=store,
                snapshot=snapshot,
                arm_index=index,
            )
    selected_arm = _select_wdl_arm(wdl_arms) if wdl_arms else None
    passed = selected_arm is not None
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
                    key: str(getattr(config, key))
                    for key in ("output_dir", "model_path", "runs_root", "telemetry_path")
                },
            },
            "provenance": provenance,
            "split": split,
            "release_wdl": release_wdl,
            "material": {
                **material,
                "model_path": str(material_model_path),
                "model_sha256": _sha256(material_model_path),
            },
            "wdl_arms": wdl_arms,
            "selected_wdl_arm": selected_arm,
            "passed": passed,
            "continuation_ranking_authorized": passed,
            "continuous_learning_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    detail = (
        f"MIHVER decoupled WDL passed · {selected_arm}"
        if passed
        else (
            "MIHVER decoupled material failed · WDL blocked"
            if not material["passed"]
            else "MIHVER decoupled WDL failed · continuation blocked"
        )
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=detail,
        pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
        pilot_stop_reason="decoupled_value_gate",
        pilot_stop_detail=detail,
        pilot_reasons=tuple(material["reasons"])
        + tuple(reason for arm in wdl_arms.values() for reason in arm["reasons"]),
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument(
        "--wdl-sampling-mode",
        choices=("outcome", "mixed", "natural"),
        default="outcome",
    )
    arguments = parser.parse_args(argv)
    print(
        run_decoupled_value_transfer(
            DecoupledValueTransferConfig(
                output_dir=arguments.output_dir,
                model_path=arguments.model,
                runs_root=arguments.runs_root,
                telemetry_path=arguments.telemetry,
                wdl_sampling_mode=arguments.wdl_sampling_mode,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
