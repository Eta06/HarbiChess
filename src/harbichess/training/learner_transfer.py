"""Run the KOPRU learner only after replay and teacher qualification gates pass."""

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

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import (
    CheckpointStatus,
    HistoryPoint,
    PilotStatus,
    RunMode,
    SnapshotStore,
)
from harbichess.evaluation.model_quality import ModelQualityMetrics, evaluate_model_quality
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.diagnostics import run_tactical_sweep
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.value_oracle import (
    DeterministicTacticalOracle,
    OracleValueEvaluator,
    TacticalOracleConfig,
)
from harbichess.training.learner import LearnerConfig, LearnerSnapshot, MLXLearner
from harbichess.training.pilot import PilotConfig, run_sanity_pilot


@dataclass(frozen=True, slots=True)
class LearnerTransferConfig:
    replay_run_result: Path
    teacher_audit_result: Path
    output_dir: Path
    replay_alignment_result: Path | None = None
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    steps: int = 200
    batch_size: int = 64
    learning_rate: float = 0.0002
    validation_interval_steps: int = 10
    early_stopping_patience: int = 12
    seed: int = 2026082803
    tactical_budgets: tuple[int, ...] = (64, 512)
    tactical_workers: int = 8
    minimum_policy_improvement: float = 0.02
    minimum_top_action_agreement_ratio: float = 1.0
    maximum_value_loss_ratio: float = 1.02
    maximum_ece_regression: float = 0.02

    def __post_init__(self) -> None:
        if (
            min(
                self.steps,
                self.batch_size,
                self.validation_interval_steps,
                self.early_stopping_patience,
                self.tactical_workers,
            )
            <= 0
            or self.learning_rate <= 0
            or not self.tactical_budgets
            or any(budget <= 0 for budget in self.tactical_budgets)
            or not 0.0 <= self.minimum_policy_improvement < 1.0
            or not 0.0 <= self.minimum_top_action_agreement_ratio <= 1.0
            or self.maximum_value_loss_ratio < 1.0
            or self.maximum_ece_regression < 0.0
        ):
            raise ValueError("learner transfer configuration is invalid")


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True)
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


def _config_payload(config: LearnerTransferConfig) -> dict[str, object]:
    payload = asdict(config)
    for name in (
        "replay_run_result",
        "teacher_audit_result",
        "output_dir",
        "replay_alignment_result",
        "telemetry_path",
    ):
        value = getattr(config, name)
        payload[name] = str(value) if value is not None else None
    return payload


def _network_config(payload: dict[str, object]) -> NetworkConfig:
    return NetworkConfig(
        trunk_channels=int(payload["trunk_channels"]),
        residual_blocks=int(payload["residual_blocks"]),
        policy_channels=int(payload["policy_channels"]),
        value_channels=int(payload["value_channels"]),
        value_hidden=int(payload["value_hidden"]),
    )


def _tactical_metrics(
    network: HarbiChessNetwork,
    *,
    network_config: NetworkConfig,
    budgets: tuple[int, ...],
    workers: int,
    seed: int,
) -> dict[str, object]:
    rules = PythonChessRules()
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network),
        max_batch_size=max(8, min(64, workers * 2)),
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
            budgets=budgets,
            workers=workers,
            seed=seed,
        )
    finally:
        batcher.close()


def _tactical_solved(payload: dict[str, object]) -> tuple[int, tuple[int, ...]]:
    raw = int(payload["raw"]["solved"])
    searched = tuple(int(row["solved"]) for row in payload["budgets"])
    return raw, searched


def _candidate_reasons(
    quality: ModelQualityMetrics,
    tactical: dict[str, object],
    *,
    baseline_quality: ModelQualityMetrics,
    baseline_tactical: dict[str, object],
    config: LearnerTransferConfig,
    maximum_gradient_norm: float,
) -> tuple[str, ...]:
    reasons = []
    if quality.teacher_policy_cross_entropy > baseline_quality.teacher_policy_cross_entropy * (
        1.0 - config.minimum_policy_improvement
    ):
        reasons.append("teacher-policy imitation did not improve by 2%")
    if quality.teacher_top_action_agreement < (
        baseline_quality.teacher_top_action_agreement * config.minimum_top_action_agreement_ratio
    ):
        reasons.append("teacher top-action agreement regressed")
    if quality.value_cross_entropy > (
        baseline_quality.value_cross_entropy * config.maximum_value_loss_ratio
    ):
        reasons.append("known-outcome WDL cross-entropy regressed")
    if quality.expected_score_ece > (
        baseline_quality.expected_score_ece + config.maximum_ece_regression
    ):
        reasons.append("expected-score calibration regressed")
    baseline_raw, baseline_search = _tactical_solved(baseline_tactical)
    candidate_raw, candidate_search = _tactical_solved(tactical)
    if candidate_raw < baseline_raw:
        reasons.append("raw-policy tactical solve count regressed")
    if any(
        candidate < baseline
        for candidate, baseline in zip(candidate_search, baseline_search, strict=True)
    ):
        reasons.append("search tactical solve count regressed")
    if not math.isfinite(maximum_gradient_norm) or maximum_gradient_norm > 5.0:
        reasons.append("gradient safety limit was exceeded")
    return tuple(reasons)


def _select_validation_snapshots(
    snapshots: list[tuple[int, float, LearnerSnapshot]],
    *,
    maximum: int,
) -> tuple[tuple[int, float, LearnerSnapshot], ...]:
    """Select evenly spaced validation snapshots without looking at their metrics."""

    if maximum <= 0:
        raise ValueError("maximum validation snapshots must be positive")
    if len(snapshots) <= maximum:
        return tuple(snapshots)
    indices = tuple(round(index * (len(snapshots) - 1) / (maximum - 1)) for index in range(maximum))
    return tuple(snapshots[index] for index in indices)


def _validate_replay_alignment(alignment: dict[str, object], *, replay_run_result: Path) -> None:
    gate = alignment.get("gate", {})
    if not isinstance(gate, dict) or not gate.get("passed"):
        raise ValueError("learner transfer requires a passed fresh replay alignment audit")
    config = alignment.get("config", {})
    if not isinstance(config, dict) or Path(str(config.get("run_result", ""))) != (
        replay_run_result
    ):
        raise ValueError("replay alignment audit does not match the learner replay")


def run_learner_transfer(config: LearnerTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"learner transfer output already exists: {config.output_dir}")
    replay_result = json.loads(config.replay_run_result.read_text(encoding="utf-8"))
    teacher_audit = json.loads(config.teacher_audit_result.read_text(encoding="utf-8"))
    if config.replay_alignment_result is None:
        raise ValueError("learner transfer requires a fresh replay alignment artifact")
    replay_alignment = json.loads(config.replay_alignment_result.read_text(encoding="utf-8"))
    if replay_result.get("mode") != "generation_only" or not replay_result.get("passed"):
        raise ValueError("learner transfer requires a qualified generation-only replay")
    if 64 not in teacher_audit.get("gate", {}).get("qualified_oracle_budgets", []):
        raise ValueError("learner transfer requires the fresh 64-simulation teacher audit")
    if not teacher_audit.get("gate", {}).get("bootstrap_teacher_qualified"):
        raise ValueError("fresh teacher audit did not pass its tactical/strength gate")
    _validate_replay_alignment(
        replay_alignment,
        replay_run_result=config.replay_run_result,
    )

    train_path = config.replay_run_result.parent / "replay" / "train-00000.jsonl.gz"
    validation_path = config.replay_run_result.parent / "replay" / "validation-00000.jsonl.gz"
    train = read_shard(train_path).records
    validation = read_shard(validation_path).records
    if {record.game_id for record in train} & {record.game_id for record in validation}:
        raise ValueError("learner transfer detected train/validation game leakage")

    network_config = _network_config(replay_result["config"])
    baseline_path = Path(replay_result["baseline"]["path"])
    baseline = HarbiChessNetwork(network_config)
    baseline.load_weights(str(baseline_path))
    baseline_quality = evaluate_model_quality(baseline, validation)
    baseline_tactical = _tactical_metrics(
        baseline,
        network_config=network_config,
        budgets=config.tactical_budgets,
        workers=config.tactical_workers,
        seed=config.seed,
    )

    network = HarbiChessNetwork(network_config)
    network.load_weights(str(baseline_path))
    learner = MLXLearner(
        network,
        config=LearnerConfig(
            learning_rate=config.learning_rate,
            weight_decay=0.0,
            max_gradient_norm=5.0,
        ),
    )
    store = SnapshotStore(config.telemetry_path)
    snapshot = store.read()
    started = time.perf_counter()
    validation_snapshots: list[tuple[int, float, LearnerSnapshot]] = []

    def on_step(metric, validation_loss) -> None:
        nonlocal snapshot
        if validation_loss is None:
            return
        validation_snapshots.append((metric.step, validation_loss, learner.snapshot()))
        elapsed = time.perf_counter() - started
        point = HistoryPoint(
            training_step=metric.step,
            training_elapsed_seconds=elapsed,
            lifetime_games=snapshot.lifetime_games,
            total_loss=metric.total_loss,
            elo_delta=None,
            elo_low=None,
            elo_high=None,
            games_per_hour=snapshot.games_per_hour,
            positions_per_second=metric.step * config.batch_size / max(elapsed, 1e-9),
            policy_loss=metric.policy_loss,
            value_loss=metric.value_loss,
            validation_loss=validation_loss,
        )
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.TRAINING,
            mode_detail=f"KOPRU learner transfer · {metric.step}/{config.steps} steps",
            session_elapsed_seconds=elapsed,
            training_elapsed_seconds=elapsed,
            training_step=metric.step,
            pilot_status=PilotStatus.TRAINING,
            pilot_steps_planned=config.steps,
            pilot_steps_completed=metric.step,
            policy_loss=metric.policy_loss,
            value_loss=metric.value_loss,
            total_loss=metric.total_loss,
            history=(*snapshot.history, point)[-240:],
        )
        store.write_atomic(snapshot)

    report = run_sanity_pilot(
        learner,
        train,
        validation,
        config=PilotConfig(
            steps=config.steps,
            batch_size=config.batch_size,
            minimum_train_improvement=0.0,
            maximum_validation_ratio=100.0,
            validation_interval_steps=config.validation_interval_steps,
            early_stopping_patience=config.early_stopping_patience,
            minimum_validation_delta=1e-3,
            maximum_value_validation_ratio=config.maximum_value_loss_ratio,
            checkpoint_interval_steps=config.validation_interval_steps,
            maximum_validation_checkpoints=8,
            seed=config.seed,
        ),
        on_step=on_step,
    )

    candidates = []
    eligible: list[tuple[float, int, LearnerSnapshot]] = []
    selected_snapshots = _select_validation_snapshots(validation_snapshots, maximum=8)
    for candidate_step, validation_loss, candidate_snapshot in selected_snapshots:
        learner.restore(candidate_snapshot)
        quality = evaluate_model_quality(network, validation)
        tactical = _tactical_metrics(
            network,
            network_config=network_config,
            budgets=config.tactical_budgets,
            workers=config.tactical_workers,
            seed=config.seed,
        )
        reasons = _candidate_reasons(
            quality,
            tactical,
            baseline_quality=baseline_quality,
            baseline_tactical=baseline_tactical,
            config=config,
            maximum_gradient_norm=report.maximum_gradient_norm,
        )
        candidates.append(
            {
                "step": candidate_step,
                "validation_loss": validation_loss,
                "quality": quality.to_dict(),
                "tactical": tactical,
                "passed": not reasons,
                "reasons": reasons,
            }
        )
        if not reasons:
            eligible.append(
                (
                    quality.teacher_policy_cross_entropy,
                    candidate_step,
                    candidate_snapshot,
                )
            )

    checkpoint = None
    if eligible:
        _, selected_step, selected_snapshot = min(eligible)
        learner.restore(selected_snapshot)
        checkpoint_dir = config.output_dir / f"candidate-step-{selected_step:06d}"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "model.safetensors"
        temporary = checkpoint_dir / ".model.tmp.safetensors"
        network.save_weights(str(temporary))
        os.replace(temporary, checkpoint_path)
        checkpoint = {
            "step": selected_step,
            "path": str(checkpoint_path),
            "model_sha256": _sha256(checkpoint_path),
            "arena_authorized": True,
            "promotion_authorized": False,
        }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = config.output_dir / "result.json"
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": _source_commit(),
        "passed": checkpoint is not None,
        "config": _config_payload(config),
        "baseline": {
            "path": str(baseline_path),
            "model_sha256": replay_result["baseline"]["model_sha256"],
            "quality": baseline_quality.to_dict(),
            "tactical": baseline_tactical,
        },
        "pilot": {
            "attempted_steps": report.attempted_steps,
            "restored_step": report.steps,
            "initial_validation_loss": report.initial_validation_loss,
            "initial_validation_value_loss": report.initial_validation_value_loss,
            "last_validation_step": report.last_validation_step,
            "last_validation_loss": report.last_validation_loss,
            "last_validation_value_loss": report.last_validation_value_loss,
            "maximum_gradient_norm": report.maximum_gradient_norm,
            "maximum_unclipped_gradient_norm": report.maximum_unclipped_gradient_norm,
            "stopped_early": report.stopped_early,
            "stop_reason": report.stop_reason,
        },
        "candidates": candidates,
        "checkpoint": checkpoint,
        "arena_authorized": checkpoint is not None,
        "promotion_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(result_path, payload)
    all_reasons = tuple(
        sorted({reason for candidate in candidates for reason in candidate["reasons"]})
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "KOPRU learner transfer passed · arena may be evaluated"
            if checkpoint
            else "KOPRU learner transfer failed · arena remains blocked"
        ),
        pilot_status=PilotStatus.PASSED if checkpoint else PilotStatus.FAILED,
        pilot_steps_attempted=report.attempted_steps,
        pilot_steps_completed=report.steps,
        pilot_best_validation_step=report.best_validation_step,
        pilot_best_validation_loss=report.best_validation_loss,
        pilot_stopped_early=report.stopped_early,
        pilot_stop_reason=str(report.stop_reason),
        pilot_stop_detail=(
            "All preregistered transfer gates passed"
            if checkpoint
            else "No validation checkpoint preserved policy, value, calibration, and tactics"
        ),
        pilot_reasons=() if checkpoint else all_reasons,
        candidate_checkpoint=(f"candidate-step-{checkpoint['step']:06d}" if checkpoint else "None"),
        checkpoint_status=CheckpointStatus.VERIFIED if checkpoint else CheckpointStatus.NONE,
        checkpoint_path=checkpoint["path"] if checkpoint else "",
        checkpoint_verified=checkpoint is not None,
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-run-result", required=True, type=Path)
    parser.add_argument("--teacher-audit-result", required=True, type=Path)
    parser.add_argument("--replay-alignment-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument("--seed", type=int, default=2026082803)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    path = run_learner_transfer(
        LearnerTransferConfig(
            replay_run_result=arguments.replay_run_result,
            teacher_audit_result=arguments.teacher_audit_result,
            replay_alignment_result=arguments.replay_alignment_result,
            output_dir=arguments.output_dir,
            telemetry_path=arguments.telemetry,
            seed=arguments.seed,
        )
    )
    print(path)
    return 0 if json.loads(path.read_text(encoding="utf-8"))["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
