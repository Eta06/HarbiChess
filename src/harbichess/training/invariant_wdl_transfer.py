"""Calibrate a policy-preserving invariant value tower on corrected terminal WDL."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from harbichess.backends.invariant_value_network import HarbiChessInvariantValueNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.corrected_replay_value_transfer import (
    CorrectedReplayValueTransferConfig,
    _load_games,
    _split_games,
)
from harbichess.evaluation.deterministic_value_probe import _prepare, _quality
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.training.full_gumbel_transfer import _network, _snapshot
from harbichess.training.joint_policy_value_transfer import (
    OutcomeGameBalancedSampler,
    _parameter_hash,
    _sha256,
    _value_gate_reasons,
    _value_quality,
)


@dataclass(frozen=True, slots=True)
class InvariantWDLTransferConfig:
    output_dir: Path
    model_path: Path
    material_result: Path
    runs_root: Path = Path("artifacts/runs")
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    batch_size: int = 64
    steps: int = 400
    validation_interval: int = 20
    learning_rate: float = 5e-4
    retention_weight: float = 350.0
    seed: int = 2026083073

    def __post_init__(self) -> None:
        if (
            min(
                self.batch_size,
                self.steps,
                self.validation_interval,
                self.learning_rate,
                self.retention_weight,
                self.seed,
            )
            <= 0
            or self.steps % self.validation_interval
        ):
            raise ValueError("invariant WDL transfer configuration is invalid")


_TOWER_PREFIXES = (
    "value_tower_stem.",
    "value_tower_blocks.",
    "value_tower_hidden.",
    "value_tower_output.",
)
_ALL_INVARIANT_PREFIXES = ("invariant_value_linear.", *_TOWER_PREFIXES)


def _wdl_quality(network, inputs: mx.array, outcomes: tuple[int, ...]) -> dict[str, object]:
    _, logits = network(inputs)
    mx.eval(logits)
    return _value_quality(logits, outcomes)


def _material_gate_reasons(
    release: dict[str, float | int], candidate: dict[str, float | int]
) -> tuple[str, ...]:
    reasons = []
    if float(candidate["mse"]) > float(release["mse"]) * 0.5:
        reasons.append("deterministic material MSE no longer clears 50 percent improvement")
    if float(candidate["pearson"]) < 0.80:
        reasons.append("deterministic material Pearson regressed below 0.80")
    if float(candidate["mae"]) > 0.05:
        reasons.append("deterministic material MAE regressed above 0.05")
    return tuple(reasons)


class _TowerLearner:
    def __init__(self, network, *, learning_rate: float, retention_weight: float) -> None:
        self.network = network
        self.retention_weight = retention_weight
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.loss_and_grad = nn.value_and_grad(network, self._loss)

    def _loss(
        self,
        network,
        inputs: mx.array,
        outcomes: mx.array,
        material_targets: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        _, logits = network(inputs)
        wdl = nn.losses.cross_entropy(logits, outcomes, reduction="mean")
        probabilities = mx.softmax(logits, axis=1)
        expected = probabilities[:, 0] - probabilities[:, 2]
        material = mx.mean(mx.square(expected - material_targets))
        total = wdl + self.retention_weight * material
        return total, wdl, material

    def step(
        self, inputs: mx.array, outcomes: mx.array, material_targets: mx.array
    ) -> tuple[float, float, float, float]:
        (total, wdl, material), gradients = self.loss_and_grad(
            self.network, inputs, outcomes, material_targets
        )
        gradients, norm = optim.clip_grad_norm(gradients, 5.0)
        finite = mx.all(
            mx.stack([mx.all(mx.isfinite(array)) for _, array in tree_flatten(gradients)])
        )
        mx.eval(total, wdl, material, norm, finite, gradients)
        if not bool(finite.item()) or not all(
            math.isfinite(float(value.item())) for value in (total, wdl, material, norm)
        ):
            raise RuntimeError("invariant WDL transfer produced non-finite gradients")
        self.optimizer.update(self.network, gradients)
        mx.eval(self.network.parameters(), self.optimizer.state)
        return tuple(  # type: ignore[return-value]
            float(value.item()) for value in (total, wdl, material, norm)
        )


def _arm_gate_reasons(
    release_wdl: dict[str, object],
    release_material: dict[str, float | int],
    candidate_wdl: dict[str, object],
    candidate_material: dict[str, float | int],
) -> tuple[str, ...]:
    return (
        *_value_gate_reasons(release_wdl, candidate_wdl),
        *_material_gate_reasons(release_material, candidate_material),
    )


def _run_arm(
    label: str,
    retention_weight: float,
    base,
    train_records,
    validation_records,
    train: tuple[mx.array, mx.array],
    validation: tuple[mx.array, mx.array],
    *,
    config: InvariantWDLTransferConfig,
    release_wdl: dict[str, object],
    release_material: dict[str, float | int],
    store: SnapshotStore,
    snapshot,
    arm_index: int,
) -> tuple[dict[str, object], object]:
    network = HarbiChessInvariantValueNetwork.from_base(base)
    network.load_weights(str(config.model_path))
    network.freeze_to_value_tower()
    anchor_hash_before = _parameter_hash(network, excluded_prefixes=_TOWER_PREFIXES)
    release_hash_before = _parameter_hash(network, excluded_prefixes=_ALL_INVARIANT_PREFIXES)
    sampler = OutcomeGameBalancedSampler(train_records, seed=config.seed)
    train_outcomes = tuple(int(record.outcome_value) for record in train_records)
    validation_outcomes = tuple(int(record.outcome_value) for record in validation_records)
    train_labels = mx.array(
        [{1: 0, 0: 1, -1: 2}[outcome] for outcome in train_outcomes], dtype=mx.int32
    )
    learner = _TowerLearner(
        network,
        learning_rate=config.learning_rate,
        retention_weight=retention_weight,
    )
    baseline = {
        "wdl": _wdl_quality(network, validation[0], validation_outcomes),
        "material": _quality(network, *validation),
    }
    best_diagnostic = (float(baseline["wdl"]["macro_cross_entropy"]), 0, _snapshot(network))
    eligible = []
    curve = []
    maximum_gradient_norm = 0.0
    for step in range(1, config.steps + 1):
        indices = sampler.sample_indices(config.batch_size)
        rows = mx.array(indices, dtype=mx.int32)
        total, wdl_loss, material_loss, norm = learner.step(
            mx.take(train[0], rows, axis=0),
            mx.take(train_labels, rows, axis=0),
            mx.take(train[1], rows, axis=0),
        )
        maximum_gradient_norm = max(maximum_gradient_norm, norm)
        if step % config.validation_interval:
            continue
        validation_quality = {
            "wdl": _wdl_quality(network, validation[0], validation_outcomes),
            "material": _quality(network, *validation),
        }
        train_quality = {
            "wdl": _wdl_quality(network, train[0], train_outcomes),
            "material": _quality(network, *train),
        }
        reasons = _arm_gate_reasons(
            release_wdl,
            release_material,
            validation_quality["wdl"],
            validation_quality["material"],
        )
        macro = float(validation_quality["wdl"]["macro_cross_entropy"])
        weights = _snapshot(network)
        if macro < best_diagnostic[0]:
            best_diagnostic = (macro, step, weights)
        if not reasons:
            eligible.append((macro, step, weights))
        curve.append(
            {
                "step": step,
                "batch_total_loss": total,
                "batch_wdl_loss": wdl_loss,
                "batch_material_loss": material_loss,
                "train": train_quality,
                "validation": validation_quality,
                "gate_reasons": reasons,
            }
        )
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode_detail=f"MIHVER WDL calibration · {label} · {step}/{config.steps}",
            pilot_steps_completed=arm_index * config.steps + step,
            training_step=arm_index * config.steps + step,
            value_loss=float(validation_quality["wdl"]["cross_entropy"]),
        )
        store.write_atomic(snapshot)
    selected_macro, selected_step, selected_weights = (
        min(eligible, key=lambda row: (row[0], row[1])) if eligible else best_diagnostic
    )
    network.load_weights(list(selected_weights))
    selected = {
        "wdl": _wdl_quality(network, validation[0], validation_outcomes),
        "material": _quality(network, *validation),
    }
    reasons = list(
        _arm_gate_reasons(
            release_wdl,
            release_material,
            selected["wdl"],
            selected["material"],
        )
    )
    anchor_hash_after = _parameter_hash(network, excluded_prefixes=_TOWER_PREFIXES)
    release_hash_after = _parameter_hash(network, excluded_prefixes=_ALL_INVARIANT_PREFIXES)
    if anchor_hash_before != anchor_hash_after:
        reasons.append("training changed the frozen material anchor")
    if release_hash_before != release_hash_after:
        reasons.append("training changed a frozen release parameter")
    model_path = config.output_dir / "arms" / label / "model.safetensors"
    model_path.parent.mkdir(parents=True)
    temporary = model_path.with_name(".model.tmp.safetensors")
    network.save_weights(str(temporary))
    os.replace(temporary, model_path)
    return {
        "label": label,
        "retention_weight": retention_weight,
        "baseline": baseline,
        "selected_step": selected_step,
        "selected_macro_wdl_ce": selected_macro,
        "selected": selected,
        "passed": not reasons and bool(eligible),
        "reasons": reasons,
        "eligible_checkpoints": [step for _, step, _ in eligible],
        "maximum_gradient_norm": maximum_gradient_norm,
        "anchor_hash_before": anchor_hash_before,
        "anchor_hash_after": anchor_hash_after,
        "release_hash_before": release_hash_before,
        "release_hash_after": release_hash_after,
        "curve": curve,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
    }, snapshot


def _select_arm(arms: dict[str, dict[str, object]]) -> str | None:
    passed = [label for label, arm in arms.items() if arm["passed"]]
    if not passed:
        return None
    if len(passed) == 1:
        return passed[0]
    plain = float(arms["tower-wdl"]["selected_macro_wdl_ce"])
    retained = float(arms["tower-wdl-retained"]["selected_macro_wdl_ce"])
    return "tower-wdl" if plain <= retained + 0.01 else "tower-wdl-retained"


def run_invariant_wdl_transfer(config: InvariantWDLTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"invariant WDL output exists: {config.output_dir}")
    material_result = json.loads(config.material_result.read_text())
    if not material_result.get("passed") or material_result.get("selected_arm") != "global-linear":
        raise ValueError("invariant WDL transfer requires the qualified global material arm")
    if str(config.model_path) != material_result["arms"]["global-linear"]["model_path"]:
        raise ValueError("invariant WDL model path does not match qualified material provenance")
    pool_config = CorrectedReplayValueTransferConfig(
        output_dir=config.output_dir,
        model_path=Path("artifacts/runs/kopru-qualified-replay-20260828-01/baseline/model.safetensors"),
        runs_root=config.runs_root,
    )
    games, provenance = _load_games(pool_config)
    train_records, validation_records, split = _split_games(games, seed=pool_config.seed)
    rules = PythonChessRules()
    train = _prepare(train_records, rules)
    validation = _prepare(validation_records, rules)
    base = _network()
    base.load_weights(str(pool_config.model_path))
    release_wdl = _wdl_quality(
        base,
        validation[0],
        tuple(int(record.outcome_value) for record in validation_records),
    )
    release_material = _quality(base, *validation)
    config.output_dir.mkdir(parents=True)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="MIHVER corrected terminal WDL calibration",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.steps * 2,
        pilot_steps_completed=0,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    arms = {}
    for index, (label, weight) in enumerate(
        (("tower-wdl", 0.0), ("tower-wdl-retained", config.retention_weight))
    ):
        arms[label], snapshot = _run_arm(
            label,
            weight,
            base,
            train_records,
            validation_records,
            train,
            validation,
            config=config,
            release_wdl=release_wdl,
            release_material=release_material,
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
                    for key in (
                        "output_dir",
                        "model_path",
                        "material_result",
                        "runs_root",
                        "telemetry_path",
                    )
                },
            },
            "provenance": provenance,
            "split": split,
            "release": {"wdl": release_wdl, "material": release_material},
            "arms": arms,
            "selected_arm": selected_arm,
            "passed": passed,
            "continuation_ranking_authorized": passed,
            "continuous_learning_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    detail = (
        f"MIHVER WDL gate passed · {selected_arm}"
        if passed
        else "MIHVER WDL gate failed · continuation blocked"
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=detail,
        pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
        pilot_stop_reason="invariant_wdl_calibration_gate",
        pilot_stop_detail=detail,
        pilot_reasons=tuple(
            reason for arm in arms.values() for reason in arm["reasons"]
        ),
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--material-result", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    print(
        run_invariant_wdl_transfer(
            InvariantWDLTransferConfig(
                output_dir=arguments.output_dir,
                model_path=arguments.model,
                material_result=arguments.material_result,
                runs_root=arguments.runs_root,
                telemetry_path=arguments.telemetry,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
