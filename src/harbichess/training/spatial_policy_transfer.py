"""Train-only qualification for the YAPI spatial policy representation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.backends.spatial_policy_network import HarbiChessSpatialPolicyNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.teacher_qualification import _atomic_json, _source_commit
from harbichess.replay.shard import read_shard
from harbichess.training.batch import GameBalancedSampler
from harbichess.training.policy_projection import _target_entropy
from harbichess.training.uncertainty_policy_transfer import (
    PreparedPolicyData,
    _network_config,
    _policy_quality,
    _prepare_data,
)


@dataclass(frozen=True, slots=True)
class SpatialPolicyTransferConfig:
    policy_target_result: Path
    dataset_result: Path
    run_result: Path
    train_shard: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    learning_rate: float = 1e-3
    batch_size: int = 16
    steps: int = 480
    seed: int = 2026082841
    max_gradient_norm: float = 5.0
    bootstrap_samples: int = 2_000
    minimum_gap_fraction: float = 0.20
    minimum_teacher_spearman: float = 0.35
    maximum_harmful_ratio: float = 0.10
    maximum_verified_regret: float = 0.10
    minimum_best_action_coverage: float = 0.80

    def __post_init__(self) -> None:
        if (
            min(self.batch_size, self.steps, self.seed, self.bootstrap_samples) <= 0
            or self.learning_rate <= 0
            or self.max_gradient_norm <= 0
            or not 0 <= self.minimum_gap_fraction <= 1
            or not -1 <= self.minimum_teacher_spearman <= 1
            or not 0 <= self.maximum_harmful_ratio <= 1
            or self.maximum_verified_regret < 0
            or not 0 <= self.minimum_best_action_coverage <= 1
        ):
            raise ValueError("spatial policy transfer configuration is invalid")


class SpatialPolicyLearner:
    def __init__(
        self,
        network: HarbiChessSpatialPolicyNetwork,
        *,
        learning_rate: float,
        max_gradient_norm: float,
    ) -> None:
        self.head = network.spatial_policy_adapter
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
        self.max_gradient_norm = max_gradient_norm
        self._loss_and_grad = nn.value_and_grad(self.head, self._loss)

    def _loss(
        self,
        trunk: mx.array,
        base_logits: mx.array,
        targets: mx.array,
        legal_masks: mx.array,
    ) -> mx.array:
        logits = mx.where(
            legal_masks,
            base_logits + self.head(trunk),
            mx.array(-1e9),
        )
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def train_step(self, batch: tuple[mx.array, ...]) -> tuple[float, float]:
        loss, gradients = self._loss_and_grad(*batch)
        gradients, raw_norm = optim.clip_grad_norm(gradients, self.max_gradient_norm)
        finite = mx.all(
            mx.stack([mx.all(mx.isfinite(value)) for _, value in tree_flatten(gradients)])
        )
        mx.eval(loss, raw_norm, finite, gradients)
        loss_value = float(loss.item())
        norm_value = float(raw_norm.item())
        if not bool(finite.item()) or not all(map(math.isfinite, (loss_value, norm_value))):
            raise RuntimeError("spatial policy loss or gradients became non-finite")
        self.optimizer.update(self.head, gradients)
        mx.eval(self.head.parameters(), self.optimizer.state)
        return loss_value, norm_value


def _batch(
    data: PreparedPolicyData, trunk: mx.array, indices: tuple[int, ...]
) -> tuple[mx.array, ...]:
    rows = mx.array(indices, dtype=mx.int32)
    return tuple(
        mx.take(array, rows, axis=0)
        for array in (trunk, data.base_logits, data.targets, data.legal_masks)
    )


def _gate_reasons(
    quality: Mapping[str, object],
    *,
    gap_fraction: float,
    maximum_gradient_norm: float,
    config: SpatialPolicyTransferConfig,
) -> tuple[str, ...]:
    reasons = []
    if gap_fraction < config.minimum_gap_fraction:
        reasons.append("reducible policy KL gap closure is below 20%")
    if float(quality["mean_teacher_policy_spearman"]) < config.minimum_teacher_spearman:
        reasons.append("teacher-policy Spearman is below 0.35")
    if float(quality["verified_delta_95_interval"][0]) <= 0:
        reasons.append("verified-improvement interval is not positive")
    if float(quality["harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("harmful-action ratio exceeds 10%")
    if float(quality["mean_verified_regret"]) > config.maximum_verified_regret:
        reasons.append("mean verified regret exceeds 0.10")
    if float(quality["best_action_coverage_top_16"]) < config.minimum_best_action_coverage:
        reasons.append("top-16 best-action coverage is below 80%")
    if not math.isfinite(maximum_gradient_norm) or maximum_gradient_norm > (
        config.max_gradient_norm
    ):
        reasons.append("gradient safety limit was exceeded")
    return tuple(reasons)


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_spatial_policy_transfer(config: SpatialPolicyTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"spatial policy output exists: {config.output_dir}")
    target = json.loads(config.policy_target_result.read_text(encoding="utf-8"))
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if not target.get("gate", {}).get("learner_transfer_authorized"):
        raise ValueError("spatial policy transfer requires a qualified target")

    network_config = _network_config(run["config"])
    baseline_path = Path(run["baseline"]["path"])
    base = HarbiChessNetwork(network_config)
    base.load_weights(str(baseline_path))
    network = HarbiChessSpatialPolicyNetwork.from_base(base)
    rules = PythonChessRules()
    train_records = read_shard(config.train_shard, rules=rules).records
    data = _prepare_data(
        train_records,
        target["rows"]["train"],
        dataset["rows"]["train"],
        network,
        explicit_targets=True,
    )
    trunk, base_logits, base_wdl = network.frozen_spatial_features(data.inputs)
    mx.eval(trunk, base_logits, base_wdl)
    mx.random.seed(config.seed)
    learner = SpatialPolicyLearner(
        network,
        learning_rate=config.learning_rate,
        max_gradient_norm=config.max_gradient_norm,
    )
    sampler = GameBalancedSampler(data.records, seed=config.seed)
    store = SnapshotStore(config.telemetry_path)
    dashboard = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail=f"YAPI spatial policy fit · 0/{config.steps}",
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.steps,
        pilot_steps_completed=0,
    )
    store.write_atomic(dashboard)
    started = time.perf_counter()
    maximum_gradient_norm = 0.0
    for step in range(1, config.steps + 1):
        _, norm = learner.train_step(
            _batch(data, trunk, sampler.sample_indices(config.batch_size))
        )
        maximum_gradient_norm = max(maximum_gradient_norm, norm)
        if step % 60 == 0 or step == config.steps:
            dashboard = replace(
                dashboard,
                updated_at=datetime.now(UTC).isoformat(),
                mode_detail=f"YAPI spatial policy fit · {step}/{config.steps}",
                pilot_steps_completed=step,
            )
            store.write_atomic(dashboard)

    baseline_quality = _policy_quality(
        base_logits,
        data,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed,
    )
    policy_logits, candidate_wdl = network(data.inputs)
    quality = _policy_quality(
        policy_logits,
        data,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed + 1,
    )
    entropy = _target_entropy(data.targets)
    baseline_ce = float(baseline_quality["uncertainty_policy_cross_entropy"])
    reducible_gap = baseline_ce - entropy
    gap_fraction = (
        baseline_ce - float(quality["uncertainty_policy_cross_entropy"])
    ) / reducible_gap
    wdl_delta_array = mx.max(mx.abs(candidate_wdl - base_wdl))
    mx.eval(wdl_delta_array)
    wdl_delta = float(wdl_delta_array.item())
    reasons = list(
        _gate_reasons(
            quality,
            gap_fraction=gap_fraction,
            maximum_gradient_norm=maximum_gradient_norm,
            config=config,
        )
    )
    if wdl_delta != 0.0:
        reasons.append("WDL logits changed")

    checkpoint = None
    if not reasons:
        checkpoint_dir = config.output_dir / "candidate"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "model.safetensors"
        temporary = checkpoint_dir / ".model.tmp.safetensors"
        network.save_weights(str(temporary))
        os.replace(temporary, checkpoint_path)
        checkpoint = {
            "path": str(checkpoint_path),
            "model_sha256": _sha256(checkpoint_path),
            "fresh_validation_authorized": True,
            "search_qualification_authorized": False,
            "arena_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
        }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = config.output_dir / "result.json"
    _atomic_json(
        result_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": _source_commit(),
            "config": {
                **asdict(config),
                **{
                    name: str(getattr(config, name))
                    for name in (
                        "policy_target_result",
                        "dataset_result",
                        "run_result",
                        "train_shard",
                        "output_dir",
                        "telemetry_path",
                    )
                },
            },
            "training": {
                "elapsed_seconds": time.perf_counter() - started,
                "maximum_gradient_norm": maximum_gradient_norm,
                "target_entropy": entropy,
                "baseline_cross_entropy": baseline_ce,
                "reducible_kl_gap": reducible_gap,
                "reducible_gap_fraction": gap_fraction,
            },
            "baseline_quality": baseline_quality,
            "quality": quality,
            "maximum_wdl_logit_delta": wdl_delta,
            "passed": checkpoint is not None,
            "reasons": reasons,
            "checkpoint": checkpoint,
            "fresh_validation_authorized": checkpoint is not None,
            "search_qualification_authorized": False,
            "arena_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
        },
    )
    dashboard = replace(
        dashboard,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "YAPI spatial policy fit passed · fresh validation authorized"
            if checkpoint
            else "YAPI spatial policy fit failed · learner blocked"
        ),
        pilot_status=PilotStatus.PASSED if checkpoint else PilotStatus.FAILED,
        pilot_steps_attempted=config.steps,
        pilot_steps_completed=config.steps,
        pilot_stop_reason="fixed_step_limit",
        pilot_stop_detail="Train fit passed" if checkpoint else "; ".join(reasons),
        promotion_ready=False,
    )
    store.write_atomic(dashboard)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-target-result", required=True, type=Path)
    parser.add_argument("--dataset-result", required=True, type=Path)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    path = run_spatial_policy_transfer(
        SpatialPolicyTransferConfig(
            policy_target_result=arguments.policy_target_result,
            dataset_result=arguments.dataset_result,
            run_result=arguments.run_result,
            train_shard=arguments.train_shard,
            output_dir=arguments.output_dir,
            telemetry_path=arguments.telemetry,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
