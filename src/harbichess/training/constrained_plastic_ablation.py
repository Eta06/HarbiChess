"""Test Pareto-aware historical/fresh gradients on the frozen plastic value path."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from harbichess.backends.plastic_value_network import (
    PLASTIC_VALUE_PREFIXES,
    HarbiChessPlasticValueNetwork,
)
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.corrected_replay_value_transfer import (
    CorrectedReplayValueTransferConfig,
    _load_games,
    _split_games,
)
from harbichess.evaluation.decoupled_value_qualification import (
    DecoupledValueQualificationConfig,
    _tactical,
    _tactical_gate,
)
from harbichess.evaluation.deterministic_value_probe import _prepare
from harbichess.evaluation.teacher_qualification import _atomic_json, select_stratified_records
from harbichess.training.continuous_policy_iteration import _continuation_floor
from harbichess.training.decoupled_value_transfer import _MixedWDLSampler
from harbichess.training.full_gumbel_transfer import _network
from harbichess.training.invariant_wdl_transfer import _wdl_quality
from harbichess.training.joint_policy_value_transfer import (
    FixedOutcomeRatioGameBalancedSampler,
    _continuation_ranking,
    _parameter_hash,
)
from harbichess.training.stable_plastic_ablation import (
    StablePlasticAblationConfig,
    _clone,
    _fresh_split,
    _load_fresh_records,
    _load_mihver,
    _save,
    _strict_wdl_reasons,
)


@dataclass(frozen=True, slots=True)
class ConstrainedPlasticAblationConfig(StablePlasticAblationConfig):
    pass


_COMBINERS = ("mgda", "pcgrad", "mean-control")


def _gradient_statistics(left, right) -> dict[str, float]:
    left_flat = dict(tree_flatten(left))
    right_flat = dict(tree_flatten(right))
    if set(left_flat) != set(right_flat):
        raise ValueError("gradient trees differ")
    dot = sum(mx.sum(left_flat[name] * right_flat[name]) for name in left_flat)
    left_sq = sum(mx.sum(value * value) for value in left_flat.values())
    right_sq = sum(mx.sum(value * value) for value in right_flat.values())
    mx.eval(dot, left_sq, right_sq)
    dot_value = float(dot.item())
    left_norm = math.sqrt(float(left_sq.item()))
    right_norm = math.sqrt(float(right_sq.item()))
    denominator = left_norm * right_norm
    return {
        "dot": dot_value,
        "historical_norm": left_norm,
        "fresh_norm": right_norm,
        "cosine": dot_value / denominator if denominator else 0.0,
    }


def _combine_gradients(left, right, *, mode: str):
    left_flat = dict(tree_flatten(left))
    right_flat = dict(tree_flatten(right))
    if set(left_flat) != set(right_flat):
        raise ValueError("gradient trees differ")
    dot = sum(mx.sum(left_flat[name] * right_flat[name]) for name in left_flat)
    left_sq = sum(mx.sum(value * value) for value in left_flat.values())
    right_sq = sum(mx.sum(value * value) for value in right_flat.values())
    mx.eval(dot, left_sq, right_sq)
    dot_value = float(dot.item())
    left_sq_value = float(left_sq.item())
    right_sq_value = float(right_sq.item())
    weights = {"historical": 0.5, "fresh": 0.5}
    combined = []
    if mode == "mgda":
        denominator = left_sq_value + right_sq_value - 2.0 * dot_value
        alpha = (
            max(0.0, min(1.0, (right_sq_value - dot_value) / denominator))
            if denominator > 1e-20
            else 0.5
        )
        weights = {"historical": alpha, "fresh": 1.0 - alpha}
        combined = [
            (name, alpha * left_flat[name] + (1.0 - alpha) * right_flat[name])
            for name in left_flat
        ]
    elif mode == "pcgrad":
        for name in left_flat:
            historical = left_flat[name]
            fresh = right_flat[name]
            if dot_value < 0.0:
                projected_historical = historical - dot_value / max(
                    right_sq_value, 1e-20
                ) * fresh
                projected_fresh = fresh - dot_value / max(
                    left_sq_value, 1e-20
                ) * historical
            else:
                projected_historical = historical
                projected_fresh = fresh
            combined.append((name, 0.5 * (projected_historical + projected_fresh)))
    elif mode == "mean-control":
        combined = [
            (name, 0.5 * (left_flat[name] + right_flat[name])) for name in left_flat
        ]
    else:
        raise ValueError(f"unknown gradient combiner: {mode}")
    return tree_unflatten(combined), weights


class _ConstrainedLearner:
    def __init__(self, network, *, learning_rate: float, mode: str) -> None:
        self.network = network
        self.mode = mode
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.loss_and_grad = nn.value_and_grad(network, self._loss)

    @staticmethod
    def _loss(network, inputs: mx.array, targets: mx.array) -> mx.array:
        return nn.losses.cross_entropy(network(inputs)[1], targets, reduction="mean")

    def step(
        self,
        historical_inputs: mx.array,
        historical_targets: mx.array,
        fresh_inputs: mx.array,
        fresh_targets: mx.array,
    ) -> dict[str, float | dict[str, float]]:
        historical_loss, historical_gradient = self.loss_and_grad(
            self.network, historical_inputs, historical_targets
        )
        fresh_loss, fresh_gradient = self.loss_and_grad(
            self.network, fresh_inputs, fresh_targets
        )
        statistics = _gradient_statistics(historical_gradient, fresh_gradient)
        gradient, weights = _combine_gradients(
            historical_gradient, fresh_gradient, mode=self.mode
        )
        gradient, norm = optim.clip_grad_norm(gradient, 5.0)
        finite = mx.all(
            mx.stack([mx.all(mx.isfinite(value)) for _, value in tree_flatten(gradient)])
        )
        mx.eval(historical_loss, fresh_loss, norm, finite, gradient)
        values = (
            float(historical_loss.item()),
            float(fresh_loss.item()),
            float(norm.item()),
        )
        if not bool(finite.item()) or not all(math.isfinite(value) for value in values):
            raise RuntimeError("constrained residual learner produced non-finite values")
        self.optimizer.update(self.network, gradient)
        mx.eval(self.network.parameters(), self.optimizer.state)
        return {
            "historical_loss": values[0],
            "fresh_loss": values[1],
            "gradient_norm": values[2],
            "gradient": statistics,
            "combination_weights": weights,
        }


def run_constrained_plastic_ablation(config: ConstrainedPlasticAblationConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"YELKEN output exists: {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    rules = PythonChessRules()
    mihver = _load_mihver(config)
    release = _network()
    release.load_weights(str(config.model_path))
    pool_config = CorrectedReplayValueTransferConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        runs_root=config.runs_root,
    )
    games, provenance = _load_games(pool_config)
    historical_train, historical_validation, historical_split = _split_games(
        games, seed=pool_config.seed
    )
    fresh_records, replay_paths = _load_fresh_records(config)
    fresh_train, fresh_validation, fresh_split = _fresh_split(
        fresh_records, seed=config.seed
    )
    historical_train_inputs, _ = _prepare(historical_train, rules)
    historical_validation_inputs, _ = _prepare(historical_validation, rules)
    fresh_train_inputs, _ = _prepare(fresh_train, rules)
    fresh_validation_inputs, _ = _prepare(fresh_validation, rules)
    historical_labels = mx.array(
        [{1: 0, 0: 1, -1: 2}[int(row.outcome_value)] for row in historical_train],
        dtype=mx.int32,
    )
    fresh_labels = mx.array(
        [{1: 0, 0: 1, -1: 2}[int(row.outcome_value)] for row in fresh_train],
        dtype=mx.int32,
    )
    historical_outcomes = tuple(int(row.outcome_value) for row in historical_validation)
    fresh_outcomes = tuple(int(row.outcome_value) for row in fresh_validation)
    ranking_records = select_stratified_records(
        historical_validation,
        rules=rules,
        count=config.ranking_positions,
        seed=2026083091,
    )
    baseline = HarbiChessPlasticValueNetwork.from_mihver(mihver)
    baseline_old = _wdl_quality(baseline, historical_validation_inputs, historical_outcomes)
    baseline_fresh = _wdl_quality(baseline, fresh_validation_inputs, fresh_outcomes)
    baseline_continuation = _continuation_ranking(
        release, baseline, ranking_records, rules=rules, depth=config.ranking_depth
    )
    tactical_config = replace(
        DecoupledValueQualificationConfig(
            output_dir=config.output_dir,
            value_result=config.value_result,
            model_path=config.model_path,
        ),
        tactical_seed=config.tactical_seed,
        search_workers=config.search_workers,
        fixed_inference_batch_size=config.fixed_inference_batch_size,
        inference_wait_seconds=config.inference_wait_seconds,
    )
    baseline_tactical = _tactical(_clone(baseline), config=tactical_config)

    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="YELKEN constrained plastic gradients",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=len(_COMBINERS) * config.steps,
        pilot_steps_completed=0,
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    completed_steps = 0
    arms = {}
    selected_arm = None
    for mode in _COMBINERS:
        mx.random.seed(config.seed)
        network = HarbiChessPlasticValueNetwork.from_mihver(mihver)
        network.freeze_to_plastic_value()
        immutable_hash = _parameter_hash(network, excluded_prefixes=PLASTIC_VALUE_PREFIXES)
        learner = _ConstrainedLearner(network, learning_rate=config.learning_rate, mode=mode)
        historical_sampler = _MixedWDLSampler(historical_train, seed=config.seed)
        fresh_sampler = FixedOutcomeRatioGameBalancedSampler(
            fresh_train,
            seed=config.seed + 1,
            outcome_counts={
                -1: config.batch_size // 8,
                0: config.batch_size // 4,
                1: config.batch_size // 8,
            },
        )
        curve = []
        accepted = None
        for step in range(1, config.steps + 1):
            half = config.batch_size // 2
            historical_rows = mx.array(historical_sampler.sample_indices(half), dtype=mx.int32)
            fresh_rows = mx.array(fresh_sampler.sample_indices(half), dtype=mx.int32)
            metrics = learner.step(
                mx.take(historical_train_inputs, historical_rows, axis=0),
                mx.take(historical_labels, historical_rows, axis=0),
                mx.take(fresh_train_inputs, fresh_rows, axis=0),
                mx.take(fresh_labels, fresh_rows, axis=0),
            )
            old_quality = _wdl_quality(network, historical_validation_inputs, historical_outcomes)
            fresh_quality = _wdl_quality(network, fresh_validation_inputs, fresh_outcomes)
            numeric_reasons = (
                *_strict_wdl_reasons(baseline_old, old_quality, label="old"),
                *_strict_wdl_reasons(baseline_fresh, fresh_quality, label="fresh"),
            )
            row = {
                "step": step,
                **metrics,
                "old_wdl": old_quality,
                "fresh_wdl": fresh_quality,
                "numeric_reasons": numeric_reasons,
            }
            curve.append(row)
            completed_steps += 1
            snapshot = replace(
                snapshot,
                updated_at=datetime.now(UTC).isoformat(),
                mode_detail=f"YELKEN {mode} · step {step}/{config.steps}",
                pilot_steps_completed=completed_steps,
                training_step=completed_steps,
                value_loss=float(metrics["historical_loss"]),
                total_loss=float(metrics["historical_loss"])
                + float(metrics["fresh_loss"]),
            )
            store.write_atomic(snapshot)
            if numeric_reasons:
                continue
            continuation = _continuation_ranking(
                release, network, ranking_records, rules=rules, depth=config.ranking_depth
            )
            tactical = _tactical(_clone(network), config=tactical_config)
            external_reasons = (
                *_continuation_floor(continuation),
                *(
                    ("continuation Spearman regressed versus MIHVER",)
                    if float(continuation["candidate_mean_spearman"])
                    < float(baseline_continuation["candidate_mean_spearman"])
                    else ()
                ),
                *(
                    ("continuation top agreement regressed versus MIHVER",)
                    if float(continuation["candidate_verified_top_agreement"])
                    < float(baseline_continuation["candidate_verified_top_agreement"])
                    else ()
                ),
                *_tactical_gate(baseline_tactical, tactical),
            )
            if external_reasons:
                row["external_reasons"] = external_reasons
                row["continuation"] = continuation
                row["tactical"] = tactical
                continue
            model_path = config.output_dir / "arms" / mode / "model.safetensors"
            accepted = {
                "step": step,
                "model_path": str(model_path),
                "model_sha256": _save(network, model_path),
                "old_wdl": old_quality,
                "fresh_wdl": fresh_quality,
                "continuation": continuation,
                "tactical": tactical,
            }
            break
        immutable_unchanged = (
            _parameter_hash(network, excluded_prefixes=PLASTIC_VALUE_PREFIXES)
            == immutable_hash
        )
        arms[mode] = {
            "passed": accepted is not None and immutable_unchanged,
            "immutable_parameters_unchanged": immutable_unchanged,
            "selected": accepted,
            "curve": curve,
        }
        if arms[mode]["passed"] and selected_arm is None:
            selected_arm = mode

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
                        "value_result",
                        "model_path",
                        "source_continuous_result",
                        "runs_root",
                        "telemetry_path",
                    )
                },
            },
            "provenance": provenance,
            "historical_split": historical_split,
            "fresh_replay_paths": replay_paths,
            "fresh_split": fresh_split,
            "baseline": {
                "old_wdl": baseline_old,
                "fresh_wdl": baseline_fresh,
                "continuation": baseline_continuation,
                "tactical": baseline_tactical,
            },
            "arms": arms,
            "selected_arm": selected_arm,
            "passed": passed,
            "fresh_continuous_pilot_authorized": passed,
            "production_generation_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    detail = (
        f"YELKEN constrained ablation passed · {selected_arm} selected"
        if passed
        else "YELKEN constrained ablation failed · fresh generation blocked"
    )
    store.write_atomic(
        replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.IDLE,
            mode_detail=detail,
            pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
            pilot_stop_reason="constrained_residual_gate",
            pilot_stop_detail=detail,
            promotion_ready=False,
        )
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--value-result", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-continuous-result", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument("--seed", type=int, default=2026083111)
    arguments = parser.parse_args(argv)
    result = run_constrained_plastic_ablation(
        ConstrainedPlasticAblationConfig(
            output_dir=arguments.output_dir,
            value_result=arguments.value_result,
            model_path=arguments.model,
            source_continuous_result=arguments.source_continuous_result,
            runs_root=arguments.runs_root,
            telemetry_path=arguments.telemetry,
            seed=arguments.seed,
        )
    )
    print(result)
    return 0 if json.loads(result.read_text(encoding="utf-8"))["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
