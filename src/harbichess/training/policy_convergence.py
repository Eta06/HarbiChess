"""Measure low-rank policy imitation convergence on train data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
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
from harbichess.training.policy_projection import (
    PolicyProjectionConfig,
    _projection_reasons,
    _target_entropy,
)
from harbichess.training.uncertainty_policy_transfer import (
    LowRankPolicyAdapter,
    PolicyAdapterLearner,
    _clone_adapter,
    _merged_network,
    _network_config,
    _prepare_data,
    _quality,
    _snapshot,
)


@dataclass(frozen=True, slots=True)
class PolicyConvergenceConfig:
    policy_target_result: Path
    dataset_result: Path
    run_result: Path
    train_shard: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    rank: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 16
    checkpoint_steps: tuple[int, ...] = (480, 960, 1920)
    seed: int = 2026082843
    max_gradient_norm: float = 5.0
    bootstrap_samples: int = 2_000
    minimum_gap_fraction: float = 0.20
    minimum_teacher_spearman: float = 0.35
    maximum_harmful_ratio: float = 0.10
    maximum_verified_regret: float = 0.10
    minimum_best_action_coverage: float = 0.80

    def __post_init__(self) -> None:
        if (
            min(self.rank, self.batch_size, self.seed, self.bootstrap_samples) <= 0
            or self.learning_rate <= 0
            or self.max_gradient_norm <= 0
            or not self.checkpoint_steps
            or self.checkpoint_steps != tuple(sorted(set(self.checkpoint_steps)))
            or self.checkpoint_steps[0] <= 0
        ):
            raise ValueError("policy convergence configuration is invalid")

    @property
    def steps(self) -> int:
        return self.checkpoint_steps[-1]

    def projection_config(self) -> PolicyProjectionConfig:
        return PolicyProjectionConfig(
            policy_target_result=self.policy_target_result,
            dataset_result=self.dataset_result,
            run_result=self.run_result,
            train_shard=self.train_shard,
            output_dir=self.output_dir,
            telemetry_path=self.telemetry_path,
            rank=self.rank,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            steps=self.steps,
            seed=self.seed,
            max_gradient_norm=self.max_gradient_norm,
            bootstrap_samples=self.bootstrap_samples,
            minimum_gap_fraction=self.minimum_gap_fraction,
            minimum_teacher_spearman=self.minimum_teacher_spearman,
            maximum_harmful_ratio=self.maximum_harmful_ratio,
            maximum_verified_regret=self.maximum_verified_regret,
            minimum_best_action_coverage=self.minimum_best_action_coverage,
        )


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_policy_convergence(config: PolicyConvergenceConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"policy convergence output exists: {config.output_dir}")
    target = json.loads(config.policy_target_result.read_text(encoding="utf-8"))
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if not target.get("gate", {}).get("learner_transfer_authorized"):
        raise ValueError("policy convergence requires a qualified target")

    network_config = _network_config(run["config"])
    baseline_path = Path(run["baseline"]["path"])
    base = HarbiChessNetwork(network_config)
    base.load_weights(str(baseline_path))
    rules = PythonChessRules()
    train_records = read_shard(config.train_shard, rules=rules).records
    data = _prepare_data(
        train_records,
        target["rows"]["train"],
        dataset["rows"]["train"],
        base,
        explicit_targets=True,
    )
    mx.random.seed(config.seed)
    feature_size = int(data.features.shape[1])
    adapter = LowRankPolicyAdapter(feature_size, config.rank)
    learner = PolicyAdapterLearner(
        adapter,
        learning_rate=config.learning_rate,
        max_gradient_norm=config.max_gradient_norm,
    )
    sampler = GameBalancedSampler(data.records, seed=config.seed)
    baseline_quality = _quality(
        adapter,
        data,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed,
        scale=0.0,
    )
    entropy = _target_entropy(data.targets)
    baseline_ce = float(baseline_quality["uncertainty_policy_cross_entropy"])
    reducible_gap = baseline_ce - entropy
    if reducible_gap <= 0:
        raise ValueError("qualified target has no positive reducible KL gap")

    store = SnapshotStore(config.telemetry_path)
    dashboard = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail=f"YAKINSAMA policy fit · 0/{config.steps}",
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.steps,
        pilot_steps_completed=0,
    )
    store.write_atomic(dashboard)
    started = time.perf_counter()
    maximum_gradient_norm = 0.0
    checkpoints = []
    projection_config = config.projection_config()
    for step in range(1, config.steps + 1):
        _, norm = learner.train_step(data.select(sampler.sample_indices(config.batch_size)))
        maximum_gradient_norm = max(maximum_gradient_norm, norm)
        if step not in config.checkpoint_steps:
            continue
        quality = _quality(
            adapter,
            data,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed + step,
        )
        gap_fraction = (
            baseline_ce - float(quality["uncertainty_policy_cross_entropy"])
        ) / reducible_gap
        reasons = _projection_reasons(
            quality,
            gap_fraction=gap_fraction,
            maximum_gradient_norm=maximum_gradient_norm,
            config=projection_config,
        )
        checkpoints.append(
            {
                "step": step,
                "reducible_gap_fraction": gap_fraction,
                "quality": quality,
                "passed": not reasons,
                "reasons": reasons,
                "weights": _snapshot(adapter),
            }
        )
        dashboard = replace(
            dashboard,
            updated_at=datetime.now(UTC).isoformat(),
            mode_detail=f"YAKINSAMA policy fit · {step}/{config.steps}",
            pilot_steps_completed=step,
        )
        store.write_atomic(dashboard)

    selected = next((row for row in checkpoints if row["passed"]), None)
    checkpoint = None
    if selected is not None:
        selected_adapter = _clone_adapter(
            feature_size,
            config.rank,
            selected["weights"],
        )
        network = _merged_network(baseline_path, network_config, selected_adapter)
        checkpoint_dir = config.output_dir / "candidate"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "model.safetensors"
        temporary = checkpoint_dir / ".model.tmp.safetensors"
        network.save_weights(str(temporary))
        os.replace(temporary, checkpoint_path)
        checkpoint = {
            "step": selected["step"],
            "path": str(checkpoint_path),
            "model_sha256": _sha256(checkpoint_path),
            "fresh_validation_authorized": True,
            "search_qualification_authorized": False,
            "arena_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
        }

    serializable_checkpoints = [
        {name: value for name, value in row.items() if name != "weights"}
        for row in checkpoints
    ]
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
            "checkpoints": serializable_checkpoints,
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
            f"YAKINSAMA passed · step {checkpoint['step']}"
            if checkpoint
            else "YAKINSAMA failed · longer fit rejected"
        ),
        pilot_status=PilotStatus.PASSED if checkpoint else PilotStatus.FAILED,
        pilot_steps_attempted=config.steps,
        pilot_steps_completed=config.steps,
        pilot_stop_reason="fixed_step_limit",
        pilot_stop_detail=(
            "Fresh validation authorized"
            if checkpoint
            else "No preregistered convergence checkpoint qualified"
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
    result = run_policy_convergence(
        PolicyConvergenceConfig(
            policy_target_result=arguments.policy_target_result,
            dataset_result=arguments.dataset_result,
            run_result=arguments.run_result,
            train_shard=arguments.train_shard,
            output_dir=arguments.output_dir,
            telemetry_path=arguments.telemetry,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
