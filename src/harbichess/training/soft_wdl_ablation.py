"""Test uncertainty-preserving WDL targets with generalized Pareto gradients."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
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
from harbichess.replay.schema import ReplayRecord
from harbichess.training.batch import GameBalancedSampler
from harbichess.training.continuous_policy_iteration import _continuation_floor
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
class SoftWDLConfig(StablePlasticAblationConfig):
    pass


_ARMS = (
    ("soft-state-generalized-mgda", True, "mgda"),
    ("soft-state-mean", True, "mean"),
    ("onehot-generalized-mgda", False, "mgda"),
)
_LABEL = {-1: 2, 0: 1, 1: 0}


def _state_key(record: ReplayRecord) -> tuple[str, tuple[str, ...]]:
    return record.root_fen, record.moves


def _soft_targets(
    historical: Sequence[ReplayRecord], fresh: Sequence[ReplayRecord]
) -> tuple[mx.array, mx.array, dict[str, int]]:
    counts: dict[tuple[str, tuple[str, ...]], Counter[int]] = defaultdict(Counter)
    for record in (*historical, *fresh):
        counts[_state_key(record)][int(record.outcome_value)] += 1

    def targets(records: Sequence[ReplayRecord]) -> mx.array:
        rows = []
        for record in records:
            outcomes = counts[_state_key(record)]
            total = sum(outcomes.values())
            rows.append(
                [
                    outcomes[1] / total,
                    outcomes[0] / total,
                    outcomes[-1] / total,
                ]
            )
        return mx.array(rows, dtype=mx.float32)

    repeated = {key: value for key, value in counts.items() if sum(value.values()) > 1}
    ambiguous = {key: value for key, value in repeated.items() if len(value) > 1}
    return targets(historical), targets(fresh), {
        "unique_fit_states": len(counts),
        "repeated_fit_states": len(repeated),
        "ambiguous_fit_states": len(ambiguous),
        "ambiguous_fit_rows": sum(sum(value.values()) for value in ambiguous.values()),
    }


def _onehot_targets(records: Sequence[ReplayRecord]) -> mx.array:
    labels = mx.array([_LABEL[int(record.outcome_value)] for record in records], dtype=mx.int32)
    return mx.eye(3)[labels]


def _pearson_loss(logits: mx.array, targets: mx.array) -> mx.array:
    probabilities = mx.softmax(logits, axis=1)
    predictions = probabilities[:, 0] - probabilities[:, 2]
    expected = targets[:, 0] - targets[:, 2]
    centered_predictions = predictions - mx.mean(predictions)
    centered_expected = expected - mx.mean(expected)
    denominator = mx.sqrt(
        mx.sum(centered_predictions * centered_predictions)
        * mx.sum(centered_expected * centered_expected)
        + 1e-12
    )
    return -mx.sum(centered_predictions * centered_expected) / denominator


def _mgda_weights(gram: list[list[float]], *, iterations: int = 100) -> list[float]:
    count = len(gram)
    weights = [1.0 / count] * count
    for _ in range(iterations):
        gradient = [
            2.0 * sum(gram[row][column] * weights[column] for column in range(count))
            for row in range(count)
        ]
        vertex = min(range(count), key=gradient.__getitem__)
        direction = [-weight for weight in weights]
        direction[vertex] += 1.0
        gram_direction = [
            sum(gram[row][column] * direction[column] for column in range(count))
            for row in range(count)
        ]
        numerator = -sum(
            weights[index] * gram_direction[index] for index in range(count)
        )
        denominator = sum(
            direction[index] * gram_direction[index] for index in range(count)
        )
        gamma = max(0.0, min(1.0, numerator / denominator)) if denominator > 1e-12 else 0.0
        weights = [
            weights[index] + gamma * direction[index] for index in range(count)
        ]
    return weights


def _combine_objectives(gradients: Sequence[object], *, mode: str):
    flattened = [dict(tree_flatten(gradient)) for gradient in gradients]
    names = tuple(flattened[0])
    if any(tuple(gradient) != names for gradient in flattened[1:]):
        raise ValueError("objective gradient trees differ")
    norms = []
    for gradient in flattened:
        squared = sum(mx.sum(value * value) for value in gradient.values())
        mx.eval(squared)
        norms.append(math.sqrt(float(squared.item())))
    normalized = [
        {name: gradient[name] / max(norm, 1e-12) for name in names}
        for gradient, norm in zip(flattened, norms, strict=True)
    ]
    gram = []
    for left in normalized:
        row = []
        for right in normalized:
            dot = sum(mx.sum(left[name] * right[name]) for name in names)
            mx.eval(dot)
            row.append(float(dot.item()))
        gram.append(row)
    weights = (
        _mgda_weights(gram) if mode == "mgda" else [1.0 / len(normalized)] * len(normalized)
    )
    combined = tree_unflatten(
        [
            (
                name,
                sum(
                    weight * gradient[name]
                    for weight, gradient in zip(weights, normalized, strict=True)
                ),
            )
            for name in names
        ]
    )
    return combined, weights, gram, norms


class _MultiObjectiveLearner:
    def __init__(self, network, *, learning_rate: float, mode: str) -> None:
        self.network = network
        self.mode = mode
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.ce_grad = nn.value_and_grad(network, self._ce)
        self.pearson_grad = nn.value_and_grad(network, self._pearson)

    @staticmethod
    def _ce(network, inputs: mx.array, targets: mx.array) -> mx.array:
        return nn.losses.cross_entropy(network(inputs)[1], targets, reduction="mean")

    @staticmethod
    def _pearson(network, inputs: mx.array, targets: mx.array) -> mx.array:
        return _pearson_loss(network(inputs)[1], targets)

    def step(self, batches: Sequence[tuple[mx.array, mx.array]]) -> dict[str, object]:
        losses = []
        gradients = []
        for index, (inputs, targets) in enumerate(batches):
            function = self.ce_grad if index % 3 != 2 else self.pearson_grad
            loss, gradient = function(self.network, inputs, targets)
            losses.append(loss)
            gradients.append(gradient)
        combined, weights, gram, norms = _combine_objectives(gradients, mode=self.mode)
        combined, norm = optim.clip_grad_norm(combined, 5.0)
        finite = mx.all(
            mx.stack([mx.all(mx.isfinite(value)) for _, value in tree_flatten(combined)])
        )
        mx.eval(*losses, norm, finite, combined)
        loss_values = [float(loss.item()) for loss in losses]
        if not bool(finite.item()) or not all(math.isfinite(value) for value in loss_values):
            raise RuntimeError("soft WDL learner produced non-finite values")
        self.optimizer.update(self.network, combined)
        mx.eval(self.network.parameters(), self.optimizer.state)
        return {
            "losses": loss_values,
            "weights": weights,
            "gram": gram,
            "objective_gradient_norms": norms,
            "combined_gradient_norm": float(norm.item()),
        }


def _take(inputs, targets, indices):
    rows = mx.array(indices, dtype=mx.int32)
    return mx.take(inputs, rows, axis=0), mx.take(targets, rows, axis=0)


def run_soft_wdl_ablation(config: SoftWDLConfig) -> Path:
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
    old_train, old_validation, old_split = _split_games(games, seed=pool_config.seed)
    fresh_records, replay_paths = _load_fresh_records(config)
    fresh_train, fresh_validation, fresh_split = _fresh_split(fresh_records, seed=config.seed)
    old_inputs, _ = _prepare(old_train, rules)
    fresh_inputs, _ = _prepare(fresh_train, rules)
    old_validation_inputs, _ = _prepare(old_validation, rules)
    fresh_validation_inputs, _ = _prepare(fresh_validation, rules)
    old_soft, fresh_soft, aggregation = _soft_targets(old_train, fresh_train)
    old_onehot = _onehot_targets(old_train)
    fresh_onehot = _onehot_targets(fresh_train)
    old_outcomes = tuple(int(row.outcome_value) for row in old_validation)
    fresh_outcomes = tuple(int(row.outcome_value) for row in fresh_validation)
    ranking_records = select_stratified_records(
        old_validation, rules=rules, count=config.ranking_positions, seed=2026083091
    )
    baseline = HarbiChessPlasticValueNetwork.from_mihver(mihver)
    baseline_old = _wdl_quality(baseline, old_validation_inputs, old_outcomes)
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
        mode_detail="YELKEN uncertainty-preserving WDL ablation",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=len(_ARMS) * config.steps,
        pilot_steps_completed=0,
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    arms = {}
    selected_arm = None
    completed = 0
    for name, use_soft, mode in _ARMS:
        mx.random.seed(config.seed)
        network = HarbiChessPlasticValueNetwork.from_mihver(mihver)
        network.freeze_to_plastic_value()
        immutable_hash = _parameter_hash(network, excluded_prefixes=PLASTIC_VALUE_PREFIXES)
        learner = _MultiObjectiveLearner(network, learning_rate=config.learning_rate, mode=mode)
        old_targets = old_soft if use_soft else old_onehot
        fresh_targets = fresh_soft if use_soft else fresh_onehot
        old_micro = GameBalancedSampler(old_train, seed=config.seed)
        old_macro = FixedOutcomeRatioGameBalancedSampler(
            old_train, seed=config.seed + 1, outcome_counts={-1: 171, 0: 170, 1: 171}
        )
        fresh_micro = GameBalancedSampler(fresh_train, seed=config.seed + 2)
        fresh_macro = FixedOutcomeRatioGameBalancedSampler(
            fresh_train, seed=config.seed + 3, outcome_counts={-1: 171, 0: 170, 1: 171}
        )
        curve = []
        accepted = None
        for step in range(1, config.steps + 1):
            old_micro_batch = _take(
                old_inputs, old_targets, old_micro.sample_indices(config.batch_size // 2)
            )
            old_macro_batch = _take(
                old_inputs, old_targets, old_macro.sample_indices(config.batch_size // 2)
            )
            fresh_micro_batch = _take(
                fresh_inputs, fresh_targets, fresh_micro.sample_indices(config.batch_size // 2)
            )
            fresh_macro_batch = _take(
                fresh_inputs, fresh_targets, fresh_macro.sample_indices(config.batch_size // 2)
            )
            metrics = learner.step(
                (
                    old_micro_batch,
                    old_macro_batch,
                    old_micro_batch,
                    fresh_micro_batch,
                    fresh_macro_batch,
                    fresh_micro_batch,
                )
            )
            old_quality = _wdl_quality(network, old_validation_inputs, old_outcomes)
            fresh_quality = _wdl_quality(network, fresh_validation_inputs, fresh_outcomes)
            reasons = (
                *_strict_wdl_reasons(baseline_old, old_quality, label="old"),
                *_strict_wdl_reasons(baseline_fresh, fresh_quality, label="fresh"),
            )
            row = {
                "step": step,
                **metrics,
                "old_wdl": old_quality,
                "fresh_wdl": fresh_quality,
                "numeric_reasons": reasons,
            }
            curve.append(row)
            completed += 1
            snapshot = replace(
                snapshot,
                updated_at=datetime.now(UTC).isoformat(),
                mode_detail=f"YELKEN {name} · step {step}/{config.steps}",
                pilot_steps_completed=completed,
                training_step=completed,
                value_loss=float(sum(metrics["losses"]) / 6),
                total_loss=float(sum(metrics["losses"]) / 6),
            )
            store.write_atomic(snapshot)
            if reasons:
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
            model_path = config.output_dir / "arms" / name / "model.safetensors"
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
        immutable = (
            _parameter_hash(network, excluded_prefixes=PLASTIC_VALUE_PREFIXES)
            == immutable_hash
        )
        arms[name] = {
            "passed": accepted is not None and immutable,
            "immutable_parameters_unchanged": immutable,
            "selected": accepted,
            "curve": curve,
        }
        if arms[name]["passed"] and selected_arm is None:
            selected_arm = name
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
            "old_split": old_split,
            "fresh_replay_paths": replay_paths,
            "fresh_split": fresh_split,
            "target_aggregation": aggregation,
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
        f"YELKEN soft WDL ablation passed · {selected_arm} selected"
        if passed
        else "YELKEN soft WDL ablation failed · fresh generation blocked"
    )
    store.write_atomic(
        replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.IDLE,
            mode_detail=detail,
            pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
            pilot_stop_reason="soft_wdl_gate",
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
    result = run_soft_wdl_ablation(
        SoftWDLConfig(
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
