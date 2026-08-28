"""Validate a frozen policy-transfer candidate on one fresh teacher set."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from harbichess.training.learner_transfer import _tactical_metrics, _tactical_solved
from harbichess.training.policy_projection import (
    PolicyProjectionConfig,
    _projection_reasons,
    _target_entropy,
)
from harbichess.training.uncertainty_policy_transfer import (
    _network_config,
    _policy_quality,
    _prepare_data,
)


@dataclass(frozen=True, slots=True)
class PolicyCandidateValidationConfig:
    convergence_result: Path
    policy_target_result: Path
    dataset_result: Path
    run_result: Path
    validation_shard: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    tactical_budgets: tuple[int, ...] = (64, 512)
    tactical_workers: int = 8
    bootstrap_samples: int = 2_000
    seed: int = 2026082851
    minimum_gap_fraction: float = 0.20
    minimum_teacher_spearman: float = 0.35
    maximum_harmful_ratio: float = 0.10
    maximum_verified_regret: float = 0.10
    minimum_best_action_coverage: float = 0.80

    def __post_init__(self) -> None:
        if (
            min(self.tactical_workers, self.bootstrap_samples, self.seed) <= 0
            or not self.tactical_budgets
            or any(budget <= 0 for budget in self.tactical_budgets)
        ):
            raise ValueError("policy candidate validation configuration is invalid")

    def projection_config(self) -> PolicyProjectionConfig:
        return PolicyProjectionConfig(
            policy_target_result=self.policy_target_result,
            dataset_result=self.dataset_result,
            run_result=self.run_result,
            train_shard=self.validation_shard,
            output_dir=self.output_dir,
            telemetry_path=self.telemetry_path,
            bootstrap_samples=self.bootstrap_samples,
            seed=self.seed,
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


def run_policy_candidate_validation(config: PolicyCandidateValidationConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"candidate validation output exists: {config.output_dir}")
    convergence = json.loads(config.convergence_result.read_text(encoding="utf-8"))
    target = json.loads(config.policy_target_result.read_text(encoding="utf-8"))
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    checkpoint = convergence.get("checkpoint")
    if not convergence.get("fresh_validation_authorized") or not isinstance(checkpoint, dict):
        raise ValueError("candidate validation requires a train-qualified checkpoint")
    if not target.get("gate", {}).get("learner_transfer_authorized"):
        raise ValueError("candidate validation requires a qualified fresh policy target")
    candidate_path = Path(checkpoint["path"])
    if _sha256(candidate_path) != checkpoint["model_sha256"]:
        raise ValueError("candidate checkpoint digest mismatch")

    network_config = _network_config(run["config"])
    baseline_path = Path(run["baseline"]["path"])
    baseline = HarbiChessNetwork(network_config)
    baseline.load_weights(str(baseline_path))
    candidate = HarbiChessNetwork(network_config)
    candidate.load_weights(str(candidate_path))
    rules = PythonChessRules()
    validation_records = read_shard(config.validation_shard, rules=rules).records
    data = _prepare_data(
        validation_records,
        target["rows"]["validation"],
        dataset["rows"]["validation"],
        baseline,
        explicit_targets=True,
    )
    baseline_policy, baseline_wdl = baseline(data.inputs)
    candidate_policy, candidate_wdl = candidate(data.inputs)
    baseline_quality = _policy_quality(
        baseline_policy,
        data,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed,
    )
    quality = _policy_quality(
        candidate_policy,
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
    wdl_delta_array = mx.max(mx.abs(candidate_wdl - baseline_wdl))
    mx.eval(wdl_delta_array)
    wdl_delta = float(wdl_delta_array.item())

    baseline_tactical = _tactical_metrics(
        baseline,
        network_config=network_config,
        budgets=config.tactical_budgets,
        workers=config.tactical_workers,
        seed=config.seed,
    )
    candidate_tactical = _tactical_metrics(
        candidate,
        network_config=network_config,
        budgets=config.tactical_budgets,
        workers=config.tactical_workers,
        seed=config.seed,
    )
    baseline_solved = _tactical_solved(baseline_tactical)
    candidate_solved = _tactical_solved(candidate_tactical)
    maximum_gradient_norm = float(convergence["training"]["maximum_gradient_norm"])
    reasons = list(
        _projection_reasons(
            quality,
            gap_fraction=gap_fraction,
            maximum_gradient_norm=maximum_gradient_norm,
            config=config.projection_config(),
        )
    )
    if wdl_delta != 0.0:
        reasons.append("WDL logits changed")
    if candidate_solved[0] < baseline_solved[0] or any(
        value < baseline_value
        for value, baseline_value in zip(
            candidate_solved[1], baseline_solved[1], strict=True
        )
    ):
        reasons.append("raw-policy or search tactical solve count regressed")
    passed = not reasons

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
                        "convergence_result",
                        "policy_target_result",
                        "dataset_result",
                        "run_result",
                        "validation_shard",
                        "output_dir",
                        "telemetry_path",
                    )
                },
            },
            "candidate": checkpoint,
            "baseline_quality": baseline_quality,
            "quality": quality,
            "target_entropy": entropy,
            "baseline_cross_entropy": baseline_ce,
            "reducible_kl_gap": reducible_gap,
            "reducible_gap_fraction": gap_fraction,
            "maximum_wdl_logit_delta": wdl_delta,
            "baseline_tactical": baseline_tactical,
            "candidate_tactical": candidate_tactical,
            "passed": passed,
            "reasons": reasons,
            "search_qualification_authorized": passed,
            "arena_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
        },
    )
    store = SnapshotStore(config.telemetry_path)
    dashboard = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "KANIT learner transfer passed · search qualification authorized"
            if passed
            else "KANIT learner transfer failed · search blocked"
        ),
        pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
        pilot_stop_reason="fresh_validation_gate",
        pilot_stop_detail="Fresh validation passed" if passed else "; ".join(reasons),
        promotion_ready=False,
    )
    store.write_atomic(dashboard)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convergence-result", required=True, type=Path)
    parser.add_argument("--policy-target-result", required=True, type=Path)
    parser.add_argument("--dataset-result", required=True, type=Path)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    result = run_policy_candidate_validation(
        PolicyCandidateValidationConfig(
            convergence_result=arguments.convergence_result,
            policy_target_result=arguments.policy_target_result,
            dataset_result=arguments.dataset_result,
            run_result=arguments.run_result,
            validation_shard=arguments.validation_shard,
            output_dir=arguments.output_dir,
            telemetry_path=arguments.telemetry,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
