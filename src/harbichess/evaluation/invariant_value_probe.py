"""Qualify policy-preserving invariant value representations on deterministic value."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx

from harbichess.backends.invariant_value_network import HarbiChessInvariantValueNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.corrected_replay_value_transfer import (
    CorrectedReplayValueTransferConfig,
    _load_games,
    _split_games,
)
from harbichess.evaluation.deterministic_value_probe import (
    _prepare,
    _ProbeLearner,
    _quality,
    _round_robin,
)
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.training.full_gumbel_transfer import _network, _snapshot
from harbichess.training.joint_policy_value_transfer import _parameter_hash, _sha256


@dataclass(frozen=True, slots=True)
class InvariantValueProbeConfig:
    output_dir: Path
    model_path: Path
    runs_root: Path = Path("artifacts/runs")
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    maximum_train_positions: int = 8192
    maximum_validation_positions: int = 4096
    batch_size: int = 64
    steps: int = 200
    validation_interval: int = 20
    learning_rate: float = 2e-3
    seed: int = 2026083061

    def __post_init__(self) -> None:
        if (
            min(
                self.maximum_train_positions,
                self.maximum_validation_positions,
                self.batch_size,
                self.steps,
                self.validation_interval,
                self.seed,
            )
            <= 0
            or self.steps % self.validation_interval
            or self.learning_rate <= 0
        ):
            raise ValueError("invariant value probe configuration is invalid")


_NEW_PREFIXES = (
    "invariant_value_linear.",
    "value_tower_stem.",
    "value_tower_blocks.",
    "value_tower_hidden.",
    "value_tower_output.",
)


def _initial_equivalence(base, target, inputs: mx.array) -> dict[str, float | bool | str]:
    sample = inputs[:64]
    base_policy, base_value = base(sample)
    target_policy, target_value = target(sample)
    policy_delta = mx.max(mx.abs(base_policy - target_policy))
    value_delta = mx.max(mx.abs(base_value - target_value))
    mx.eval(policy_delta, value_delta)
    base_hash = _parameter_hash(base)
    target_release_hash = _parameter_hash(target, excluded_prefixes=_NEW_PREFIXES)
    return {
        "maximum_policy_logit_delta": float(policy_delta.item()),
        "maximum_value_logit_delta": float(value_delta.item()),
        "base_parameter_hash": base_hash,
        "target_release_parameter_hash": target_release_hash,
        "release_parameter_hash_exact": base_hash == target_release_hash,
    }


def _run_arm(
    label: str,
    base,
    train: tuple[mx.array, mx.array],
    validation: tuple[mx.array, mx.array],
    *,
    config: InvariantValueProbeConfig,
    global_only: bool,
    store: SnapshotStore,
    snapshot,
    arm_index: int,
) -> tuple[dict[str, object], object]:
    network = HarbiChessInvariantValueNetwork.from_base(base)
    equivalence = _initial_equivalence(base, network, validation[0])
    release_hash_before = str(equivalence["target_release_parameter_hash"])
    if global_only:
        network.freeze_to_global_linear()
    else:
        network.freeze_release_parameters()
    learner = _ProbeLearner(network, learning_rate=config.learning_rate)
    baseline = _quality(network, *validation)
    best_mse = float(baseline["mse"])
    best_step = 0
    best_weights = _snapshot(network)
    curve = [{"step": 0, "validation": baseline}]
    rng = random.Random(config.seed)
    for step in range(1, config.steps + 1):
        indices = tuple(rng.randrange(train[0].shape[0]) for _ in range(config.batch_size))
        rows = mx.array(indices, dtype=mx.int32)
        loss = learner.step(mx.take(train[0], rows, axis=0), mx.take(train[1], rows, axis=0))
        if step % config.validation_interval:
            continue
        quality = _quality(network, *validation)
        curve.append({"step": step, "batch_loss": loss, "validation": quality})
        if float(quality["mse"]) < best_mse:
            best_mse = float(quality["mse"])
            best_step = step
            best_weights = _snapshot(network)
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode_detail=f"MIHVER material gate · {label} · {step}/{config.steps}",
            pilot_steps_completed=arm_index * config.steps + step,
            training_step=arm_index * config.steps + step,
            value_loss=float(quality["mse"]),
        )
        store.write_atomic(snapshot)
    network.load_weights(list(best_weights))
    selected = _quality(network, *validation)
    release_hash_after = _parameter_hash(network, excluded_prefixes=_NEW_PREFIXES)
    reasons = []
    if float(equivalence["maximum_policy_logit_delta"]) != 0.0:
        reasons.append("initial policy logits were not bitwise preserved")
    if float(equivalence["maximum_value_logit_delta"]) != 0.0:
        reasons.append("initial WDL logits were not bitwise preserved")
    if not equivalence["release_parameter_hash_exact"]:
        reasons.append("initial release parameter hash changed")
    if release_hash_before != release_hash_after:
        reasons.append("training changed a frozen release parameter")
    if float(selected["mse"]) > float(baseline["mse"]) * 0.5:
        reasons.append("held-out deterministic-value MSE did not improve by 50 percent")
    if float(selected["pearson"]) < 0.80:
        reasons.append("held-out deterministic-value Pearson is below 0.80")
    if float(selected["mae"]) > 0.05:
        reasons.append("held-out deterministic-value MAE exceeds 0.05")
    model_path = config.output_dir / "arms" / label / "model.safetensors"
    model_path.parent.mkdir(parents=True)
    temporary = model_path.with_name(".model.tmp.safetensors")
    network.save_weights(str(temporary))
    os.replace(temporary, model_path)
    return {
        "label": label,
        "equivalence": equivalence,
        "release_parameter_hash_after": release_hash_after,
        "baseline": baseline,
        "selected_step": best_step,
        "selected": selected,
        "passed": not reasons,
        "reasons": reasons,
        "curve": curve,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
    }, snapshot


def _select_arm(arms: dict[str, dict[str, object]]) -> str | None:
    global_arm = arms["global-linear"]
    tower_arm = arms["invariant-tower"]
    if not global_arm["passed"] and not tower_arm["passed"]:
        return None
    if global_arm["passed"] and not tower_arm["passed"]:
        return "global-linear"
    if tower_arm["passed"] and not global_arm["passed"]:
        return "invariant-tower"
    global_mse = float(global_arm["selected"]["mse"])  # type: ignore[index]
    tower_mse = float(tower_arm["selected"]["mse"])  # type: ignore[index]
    return "invariant-tower" if tower_mse <= global_mse * 0.8 else "global-linear"


def run_invariant_value_probe(config: InvariantValueProbeConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"invariant value probe output exists: {config.output_dir}")
    pool_config = CorrectedReplayValueTransferConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        runs_root=config.runs_root,
    )
    games, provenance = _load_games(pool_config)
    train_records, validation_records, split = _split_games(games, seed=pool_config.seed)
    train_records = _round_robin(train_records, config.maximum_train_positions)
    validation_records = _round_robin(validation_records, config.maximum_validation_positions)
    rules = PythonChessRules()
    train = _prepare(train_records, rules)
    validation = _prepare(validation_records, rules)
    base = _network()
    base.load_weights(str(config.model_path))
    config.output_dir.mkdir(parents=True)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="MIHVER invariant value material qualification",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.steps * 2,
        pilot_steps_completed=0,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    arms = {}
    for index, (label, global_only) in enumerate(
        (("global-linear", True), ("invariant-tower", False))
    ):
        arms[label], snapshot = _run_arm(
            label,
            base,
            train,
            validation,
            config=config,
            global_only=global_only,
            store=store,
            snapshot=snapshot,
            arm_index=index,
        )
    selected_arm = _select_arm(arms)
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
            "selected_train_positions": len(train_records),
            "selected_validation_positions": len(validation_records),
            "arms": arms,
            "selected_arm": selected_arm,
            "passed": passed,
            "wdl_transfer_authorized": passed,
            "continuous_learning_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    detail = (
        f"MIHVER material gate passed · {selected_arm}"
        if passed
        else "MIHVER material gate failed · WDL blocked"
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=detail,
        pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
        pilot_stop_reason="invariant_value_material_gate",
        pilot_stop_detail=detail,
        pilot_reasons=tuple(reason for arm in arms.values() for reason in arm["reasons"]),
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
    arguments = parser.parse_args(argv)
    print(
        run_invariant_value_probe(
            InvariantValueProbeConfig(
                output_dir=arguments.output_dir,
                model_path=arguments.model,
                runs_root=arguments.runs_root,
                telemetry_path=arguments.telemetry,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
