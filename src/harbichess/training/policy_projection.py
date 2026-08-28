"""Fit and train-safely project the qualified SIPER policy update."""

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

from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.teacher_qualification import _atomic_json, _source_commit
from harbichess.replay.shard import read_shard
from harbichess.training.batch import GameBalancedSampler
from harbichess.training.uncertainty_policy_transfer import (
    LowRankPolicyAdapter,
    PolicyAdapterLearner,
    _merged_network,
    _network_config,
    _prepare_data,
    _quality,
)


@dataclass(frozen=True, slots=True)
class PolicyProjectionConfig:
    policy_target_result: Path
    dataset_result: Path
    run_result: Path
    train_shard: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    rank: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 16
    steps: int = 480
    seed: int = 2026082840
    max_gradient_norm: float = 5.0
    bootstrap_samples: int = 2_000
    scales: tuple[float, ...] = tuple(index / 10 for index in range(1, 11))
    minimum_gap_fraction: float = 0.20
    minimum_teacher_spearman: float = 0.35
    maximum_harmful_ratio: float = 0.10
    maximum_verified_regret: float = 0.10
    minimum_best_action_coverage: float = 0.80

    def __post_init__(self) -> None:
        if (
            min(
                self.rank,
                self.batch_size,
                self.steps,
                self.seed,
                self.bootstrap_samples,
            )
            <= 0
            or self.learning_rate <= 0
            or self.max_gradient_norm <= 0
            or not self.scales
            or self.scales != tuple(sorted(set(self.scales)))
            or any(not 0 < scale <= 1 for scale in self.scales)
            or not 0 <= self.minimum_gap_fraction <= 1
            or not -1 <= self.minimum_teacher_spearman <= 1
            or not 0 <= self.maximum_harmful_ratio <= 1
            or self.maximum_verified_regret < 0
            or not 0 <= self.minimum_best_action_coverage <= 1
        ):
            raise ValueError("policy projection configuration is invalid")


def _target_entropy(targets: mx.array) -> float:
    clipped = mx.maximum(targets, mx.array(1e-30))
    entropy = -mx.sum(targets * mx.log(clipped)) / targets.shape[0]
    mx.eval(entropy)
    return float(entropy.item())


def _projection_reasons(
    quality: Mapping[str, object],
    *,
    gap_fraction: float,
    maximum_gradient_norm: float,
    config: PolicyProjectionConfig,
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


def run_policy_projection(config: PolicyProjectionConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"policy projection output exists: {config.output_dir}")
    target = json.loads(config.policy_target_result.read_text(encoding="utf-8"))
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if not target.get("gate", {}).get("learner_transfer_authorized"):
        raise ValueError("policy projection requires a qualified target")

    network_config = _network_config(run["config"])
    baseline_path = Path(run["baseline"]["path"])
    base = HarbiChessNetwork(network_config)
    base.load_weights(str(baseline_path))
    rules = PythonChessRules()
    train_records = read_shard(config.train_shard, rules=rules).records
    train = _prepare_data(
        train_records,
        target["rows"]["train"],
        dataset["rows"]["train"],
        base,
        explicit_targets=True,
    )
    mx.random.seed(config.seed)
    adapter = LowRankPolicyAdapter(int(train.features.shape[1]), config.rank)
    learner = PolicyAdapterLearner(
        adapter,
        learning_rate=config.learning_rate,
        max_gradient_norm=config.max_gradient_norm,
    )
    sampler = GameBalancedSampler(train.records, seed=config.seed)
    store = SnapshotStore(config.telemetry_path)
    dashboard = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail=f"FREN policy fit · 0/{config.steps}",
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.steps,
        pilot_steps_completed=0,
    )
    store.write_atomic(dashboard)
    started = time.perf_counter()
    maximum_gradient_norm = 0.0
    for step in range(1, config.steps + 1):
        _, gradient_norm = learner.train_step(
            train.select(sampler.sample_indices(config.batch_size))
        )
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        if step % 60 == 0 or step == config.steps:
            dashboard = replace(
                dashboard,
                updated_at=datetime.now(UTC).isoformat(),
                mode_detail=f"FREN policy fit · {step}/{config.steps}",
                pilot_steps_completed=step,
            )
            store.write_atomic(dashboard)

    baseline_quality = _quality(
        adapter,
        train,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed,
        scale=0.0,
    )
    baseline_ce = float(baseline_quality["uncertainty_policy_cross_entropy"])
    entropy = _target_entropy(train.targets)
    reducible_gap = baseline_ce - entropy
    if reducible_gap <= 0:
        raise ValueError("qualified target has no positive reducible KL gap")
    projections = []
    eligible = []
    for index, scale in enumerate(config.scales):
        quality = _quality(
            adapter,
            train,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed + index + 1,
            scale=scale,
        )
        gap_fraction = (
            baseline_ce - float(quality["uncertainty_policy_cross_entropy"])
        ) / reducible_gap
        reasons = _projection_reasons(
            quality,
            gap_fraction=gap_fraction,
            maximum_gradient_norm=maximum_gradient_norm,
            config=config,
        )
        row = {
            "scale": scale,
            "reducible_gap_fraction": gap_fraction,
            "quality": quality,
            "passed": not reasons,
            "reasons": reasons,
        }
        projections.append(row)
        if not reasons:
            eligible.append((scale, row))

    selected = max(eligible, default=None, key=lambda item: item[0])
    checkpoint = None
    if selected is not None:
        scale, row = selected
        network = _merged_network(
            baseline_path,
            network_config,
            adapter,
            scale=scale,
        )
        checkpoint_dir = config.output_dir / "projected-candidate"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "model.safetensors"
        temporary = checkpoint_dir / ".model.tmp.safetensors"
        network.save_weights(str(temporary))
        os.replace(temporary, checkpoint_path)
        checkpoint = {
            "scale": scale,
            "path": str(checkpoint_path),
            "model_sha256": _sha256(checkpoint_path),
            "train_projection": row,
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
            },
            "baseline_quality": baseline_quality,
            "projections": projections,
            "passed": checkpoint is not None,
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
            f"FREN projection passed · scale {checkpoint['scale']:.1f}"
            if checkpoint
            else "FREN projection failed · learner blocked"
        ),
        pilot_status=PilotStatus.PASSED if checkpoint else PilotStatus.FAILED,
        pilot_steps_attempted=config.steps,
        pilot_steps_completed=config.steps,
        pilot_stop_reason="fixed_step_limit",
        pilot_stop_detail=(
            "Fresh validation authorized"
            if checkpoint
            else "No train-safe projection scale qualified"
        ),
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
    path = run_policy_projection(
        PolicyProjectionConfig(
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
