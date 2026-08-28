"""Baseline-policy trust-region ablation for Full Gumbel transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx

from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.replay.shard import read_shard
from harbichess.training.full_gumbel_transfer import (
    FullGumbelTransferConfig,
    PolicyHead,
    PolicyHeadLearner,
    _arena,
    _network,
    _parameter_hash,
    _policy_quality,
    _prepare,
    _select_indices,
    _snapshot,
    _tactical,
    _take,
    _wdl_quality,
)


@dataclass(frozen=True, slots=True)
class PolicyAnchorAblationConfig:
    output_dir: Path
    model_path: Path
    target_result: Path
    train_shard: Path
    validation_shard: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    anchor_weights: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    learning_rate: float = 2e-4
    batch_size: int = 64
    maximum_steps: int = 240
    validation_interval: int = 20
    early_stopping_patience: int = 4
    seed: int = 2026082883

    def __post_init__(self) -> None:
        if (
            not self.anchor_weights
            or len(set(self.anchor_weights)) != len(self.anchor_weights)
            or any(not math.isfinite(weight) or weight <= 0 for weight in self.anchor_weights)
        ):
            raise ValueError("policy anchor weights must be unique and positive")
        if (
            min(
                self.batch_size,
                self.maximum_steps,
                self.validation_interval,
                self.early_stopping_patience,
                self.seed,
            )
            <= 0
        ):
            raise ValueError("policy anchor ablation counts must be positive")
        if self.maximum_steps % self.validation_interval or self.learning_rate <= 0:
            raise ValueError("policy anchor schedule is invalid")


def _masked_probabilities(logits: mx.array, masks: mx.array) -> mx.array:
    return mx.softmax(mx.where(masks, logits, mx.array(-1e9)), axis=1)


def _baseline_kl(
    baseline_logits: mx.array,
    candidate_logits: mx.array,
    masks: mx.array,
) -> float:
    baseline = mx.where(masks, baseline_logits, mx.array(-1e9))
    candidate = mx.where(masks, candidate_logits, mx.array(-1e9))
    baseline_log = baseline - mx.logsumexp(baseline, axis=1, keepdims=True)
    candidate_log = candidate - mx.logsumexp(candidate, axis=1, keepdims=True)
    value = mx.mean(mx.sum(mx.exp(baseline_log) * (baseline_log - candidate_log), axis=1))
    mx.eval(value)
    return float(value.item())


def _transfer_config(config: PolicyAnchorAblationConfig) -> FullGumbelTransferConfig:
    return FullGumbelTransferConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        target_result=config.target_result,
        train_shard=config.train_shard,
        validation_shard=config.validation_shard,
        telemetry_path=config.telemetry_path,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        maximum_steps=config.maximum_steps,
        validation_interval=config.validation_interval,
        early_stopping_patience=config.early_stopping_patience,
        seed=config.seed,
    )


def _arm_reasons(
    baseline_quality: dict[str, dict[str, float]],
    candidate_quality: dict[str, dict[str, float]],
    baseline_tactical: dict[str, object],
    candidate_tactical: dict[str, object],
    *,
    policy_kl: float,
    wdl_max_logit_delta: float,
    wdl_max_metric_delta: float,
    frozen_hash_before: str,
    frozen_hash_after: str,
) -> tuple[str, ...]:
    reasons = []
    baseline_validation = baseline_quality["validation"]
    candidate_validation = candidate_quality["validation"]
    if baseline_validation["cross_entropy"] - candidate_validation["cross_entropy"] < 0.01:
        reasons.append("validation teacher cross-entropy improvement is below 0.01")
    if baseline_validation["teacher_kl"] - candidate_validation["teacher_kl"] < 0.01:
        reasons.append("validation teacher KL improvement is below 0.01")
    if (
        candidate_validation["top_action_agreement"] - baseline_validation["top_action_agreement"]
        < 0.02
    ):
        reasons.append("validation teacher top-action gain is below two points")
    if (
        candidate_validation["top_action_agreement"]
        < candidate_quality["train"]["top_action_agreement"] - 0.15
    ):
        reasons.append("validation top-action agreement trails train by over 15 points")
    if int(candidate_tactical["raw"]["solved"]) < int(baseline_tactical["raw"]["solved"]):
        reasons.append("candidate raw tactical solve count regressed")
    baseline_cases = {
        row["case"]
        for row in baseline_tactical["budgets"][0]["cases"]  # type: ignore[index]
        if row["solved"]
    }
    candidate_cases = {
        row["case"]
        for row in candidate_tactical["budgets"][0]["cases"]  # type: ignore[index]
        if row["solved"]
    }
    if int(candidate_tactical["budgets"][0]["solved"]) < 4:  # type: ignore[index]
        reasons.append("candidate 256 Full Gumbel tactical solve count is below four")
    if baseline_cases - candidate_cases:
        reasons.append("candidate search lost a baseline-solved tactical case")
    if frozen_hash_before != frozen_hash_after:
        reasons.append("frozen non-policy parameter hash changed")
    if policy_kl > 0.10:
        reasons.append("validation baseline-policy KL exceeds 0.10")
    if wdl_max_logit_delta > 1e-7 or wdl_max_metric_delta > 1e-7:
        reasons.append("frozen WDL output or calibration metrics changed")
    return tuple(reasons)


def run_policy_anchor_ablation(config: PolicyAnchorAblationConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"policy anchor output exists: {config.output_dir}")
    target = json.loads(config.target_result.read_text(encoding="utf-8"))
    if not target.get("passed") or not target.get("learner_transfer_authorized"):
        raise ValueError("policy anchor ablation requires qualified targets")
    rules = PythonChessRules()
    train = _prepare(
        read_shard(config.train_shard, rules=rules).records,
        target["rows"]["train"],
        rules=rules,
    )
    validation = _prepare(
        read_shard(config.validation_shard, rules=rules).records,
        target["rows"]["validation"],
        rules=rules,
    )
    baseline = _network()
    baseline.load_weights(str(config.model_path))
    frozen_hash = _parameter_hash(baseline, policy=False)
    train_trunk = mx.stop_gradient(baseline._trunk(train.inputs))
    validation_trunk = mx.stop_gradient(baseline._trunk(validation.inputs))
    baseline_train_logits = baseline.policy_linear(baseline._policy_features(train_trunk))
    baseline_validation_logits = baseline.policy_linear(baseline._policy_features(validation_trunk))
    baseline_wdl_logits = baseline._value_logits(validation_trunk)
    mx.eval(
        train_trunk,
        validation_trunk,
        baseline_train_logits,
        baseline_validation_logits,
        baseline_wdl_logits,
    )
    baseline_wdl_rows = baseline_wdl_logits.tolist()
    baseline_wdl_quality = _wdl_quality(baseline_wdl_rows, validation.wdl_targets)
    baseline_train_probs = _masked_probabilities(baseline_train_logits, train.legal_masks)
    baseline_quality = {
        "train": _policy_quality(baseline_train_logits, train.targets, train.legal_masks),
        "validation": _policy_quality(
            baseline_validation_logits, validation.targets, validation.legal_masks
        ),
    }
    transfer_config = _transfer_config(config)
    baseline_tactical = _tactical(config.model_path, config=transfer_config)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="AKTAR policy-anchor ablation · starting",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.maximum_steps * len(config.anchor_weights),
        pilot_steps_completed=0,
    )
    store.write_atomic(snapshot)
    config.output_dir.mkdir(parents=True)
    arms = []
    started = time.perf_counter()
    for arm_index, anchor_weight in enumerate(config.anchor_weights):
        network = _network()
        network.load_weights(str(config.model_path))
        head = PolicyHead(network)
        learner = PolicyHeadLearner(head, learning_rate=config.learning_rate)
        blended_targets = (train.targets + anchor_weight * baseline_train_probs) / (
            1.0 + anchor_weight
        )
        mx.eval(blended_targets)
        rng = random.Random(config.seed)
        best_ce = math.inf
        best_weights = None
        stale = 0
        checkpoints = []
        stop_reason = "maximum_steps"
        maximum_gradient_norm = 0.0
        for step in range(1, config.maximum_steps + 1):
            indices = _select_indices(train.records, config.batch_size, rng)
            _loss, norm = learner.train_step(
                _take(train_trunk, indices),
                _take(blended_targets, indices),
                _take(train.legal_masks, indices),
            )
            maximum_gradient_norm = max(maximum_gradient_norm, norm)
            if step % config.validation_interval == 0:
                train_logits = head(train_trunk)
                validation_logits = head(validation_trunk)
                mx.eval(train_logits, validation_logits)
                quality = {
                    "train": _policy_quality(train_logits, train.targets, train.legal_masks),
                    "validation": _policy_quality(
                        validation_logits, validation.targets, validation.legal_masks
                    ),
                }
                checkpoints.append({"step": step, "quality": quality})
                ce = quality["validation"]["cross_entropy"]
                if ce < best_ce:
                    best_ce = ce
                    best_weights = _snapshot(head)
                    stale = 0
                else:
                    stale += 1
                snapshot = replace(
                    snapshot,
                    updated_at=datetime.now(UTC).isoformat(),
                    mode_detail=(
                        f"AKTAR anchor {anchor_weight:g} · {step}/"
                        f"{config.maximum_steps} · val CE {ce:.4f}"
                    ),
                    pilot_steps_completed=arm_index * config.maximum_steps + step,
                )
                store.write_atomic(snapshot)
                if stale >= config.early_stopping_patience:
                    stop_reason = "validation_early_stopping"
                    break
        if best_weights is None:
            raise RuntimeError("policy anchor arm produced no checkpoint")
        head.load_weights(list(best_weights))
        train_logits = head(train_trunk)
        validation_logits = head(validation_trunk)
        mx.eval(train_logits, validation_logits)
        quality = {
            "train": _policy_quality(train_logits, train.targets, train.legal_masks),
            "validation": _policy_quality(
                validation_logits, validation.targets, validation.legal_masks
            ),
        }
        policy_kl = _baseline_kl(
            baseline_validation_logits, validation_logits, validation.legal_masks
        )
        candidate_wdl_logits = network._value_logits(validation_trunk)
        mx.eval(candidate_wdl_logits)
        candidate_wdl_rows = candidate_wdl_logits.tolist()
        candidate_wdl_quality = _wdl_quality(candidate_wdl_rows, validation.wdl_targets)
        wdl_max_logit_delta = max(
            abs(float(before) - float(after))
            for before_row, after_row in zip(baseline_wdl_rows, candidate_wdl_rows, strict=True)
            for before, after in zip(before_row, after_row, strict=True)
        )
        wdl_max_metric_delta = max(
            abs(float(baseline_wdl_quality[name]) - float(candidate_wdl_quality[name]))
            for name in baseline_wdl_quality
        )
        arm_dir = config.output_dir / f"anchor-{anchor_weight:g}"
        arm_dir.mkdir()
        model_path = arm_dir / "model.safetensors"
        temporary = arm_dir / ".model.tmp.safetensors"
        network.save_weights(str(temporary))
        os.replace(temporary, model_path)
        tactical = _tactical(model_path, config=transfer_config)
        reasons = _arm_reasons(
            baseline_quality,
            quality,
            baseline_tactical,
            tactical,
            policy_kl=policy_kl,
            wdl_max_logit_delta=wdl_max_logit_delta,
            wdl_max_metric_delta=wdl_max_metric_delta,
            frozen_hash_before=frozen_hash,
            frozen_hash_after=_parameter_hash(network, policy=False),
        )
        arms.append(
            {
                "anchor_weight": anchor_weight,
                "checkpoints": checkpoints,
                "selected_validation_cross_entropy": best_ce,
                "steps_completed": checkpoints[-1]["step"],
                "stop_reason": stop_reason,
                "maximum_gradient_norm": maximum_gradient_norm,
                "quality": quality,
                "validation_baseline_policy_kl": policy_kl,
                "wdl": {
                    "baseline": baseline_wdl_quality,
                    "candidate": candidate_wdl_quality,
                    "max_logit_delta": wdl_max_logit_delta,
                    "max_metric_delta": wdl_max_metric_delta,
                },
                "tactical": tactical,
                "model_path": str(model_path),
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "passed": not reasons,
                "reasons": reasons,
            }
        )
    eligible = [arm for arm in arms if arm["passed"]]
    selected = (
        min(
            eligible,
            key=lambda arm: (
                arm["quality"]["validation"]["cross_entropy"],
                -arm["anchor_weight"],
            ),
        )
        if eligible
        else None
    )
    arena = arena_control = arena_gate = None
    candidate = None
    if selected is not None:
        candidate_dir = config.output_dir / "candidate"
        candidate_dir.mkdir()
        candidate_path = candidate_dir / "model.safetensors"
        shutil.copyfile(selected["model_path"], candidate_path)
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.EVALUATION,
            mode_detail=f"AKTAR anchor {selected['anchor_weight']:g} · fresh arena",
        )
        store.write_atomic(snapshot)
        arena, arena_control, arena_gate = _arena(
            config.model_path, candidate_path, config=transfer_config
        )
        candidate = {
            "anchor_weight": selected["anchor_weight"],
            "path": str(candidate_path),
            "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        }
    passed = selected is not None and bool(arena_gate and arena_gate["passed"])
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
                        "target_result",
                        "train_shard",
                        "validation_shard",
                        "telemetry_path",
                    )
                },
            },
            "baseline_quality": baseline_quality,
            "baseline_wdl_quality": baseline_wdl_quality,
            "baseline_tactical": baseline_tactical,
            "frozen_non_policy_hash": frozen_hash,
            "arms": arms,
            "candidate": candidate,
            "arena": arena,
            "arena_control": arena_control,
            "arena_gate": arena_gate,
            "passed": passed,
            "continuous_learning_authorized": passed,
            "generation_authorized": False,
            "promotion_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "AKTAR policy anchor passed · continuous learner authorized"
            if passed
            else "AKTAR policy anchor failed · continuous learner blocked"
        ),
        pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
        pilot_stop_reason="fixed_anchor_matrix",
        pilot_stop_detail=(
            "all transfer and arena gates passed" if passed else "no anchor passed all frozen gates"
        ),
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-result", type=Path, required=True)
    parser.add_argument("--train-shard", type=Path, required=True)
    parser.add_argument("--validation-shard", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    result = run_policy_anchor_ablation(
        PolicyAnchorAblationConfig(
            output_dir=arguments.output_dir,
            model_path=arguments.model,
            target_result=arguments.target_result,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            telemetry_path=arguments.telemetry,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
