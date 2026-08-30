"""Frozen, gated joint transfer of qualified Full Gumbel policy and WDL targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import Side
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.teacher_qualification import (
    _atomic_json,
    select_stratified_records,
)
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import ReplayShard, read_shard
from harbichess.search.value_oracle import DeterministicTacticalOracle, TacticalOracleConfig
from harbichess.training.batch import build_training_batch
from harbichess.training.full_gumbel_transfer import (
    FullGumbelTransferConfig,
    _arena,
    _network,
    _policy_quality,
    _prepare,
    _select_indices,
    _snapshot,
    _tactical,
    _take,
    _wdl_quality,
)
from harbichess.training.learner import LearnerConfig, MLXLearner, PreparedTrainingBatch
from harbichess.training.value_bootstrap import _freeze_to_value_head


@dataclass(frozen=True, slots=True)
class JointPolicyValueTransferConfig:
    output_dir: Path
    model_path: Path
    target_result: Path
    train_shard: Path
    validation_shard: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    expected_model_sha256: str = "5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca"
    warmup_learning_rate: float = 5e-4
    joint_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    warmup_steps: int = 400
    joint_steps: int = 400
    validation_interval: int = 20
    early_stopping_patience: int = 6
    ranking_positions: int = 32
    ranking_depth: int = 4
    seed: int = 2026083017
    arena_seed: int = 2026083029
    require_head_warmup_gate: bool = True

    def __post_init__(self) -> None:
        counts = (
            self.batch_size,
            self.warmup_steps,
            self.joint_steps,
            self.validation_interval,
            self.early_stopping_patience,
            self.ranking_positions,
            self.ranking_depth,
            self.seed,
            self.arena_seed,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("joint transfer counts must be positive")
        if (
            self.warmup_steps % self.validation_interval
            or self.joint_steps % self.validation_interval
        ):
            raise ValueError("joint transfer schedules must align with validation interval")
        if min(self.warmup_learning_rate, self.joint_learning_rate) <= 0 or self.weight_decay < 0:
            raise ValueError("joint transfer optimizer configuration is invalid")
        if len(self.expected_model_sha256) != 64:
            raise ValueError("joint transfer expected model hash must be SHA-256")


class OutcomeGameBalancedSampler:
    """Sample known outcomes uniformly, then games uniformly inside each outcome."""

    def __init__(self, records: tuple[ReplayRecord, ...], *, seed: int) -> None:
        self._rng = random.Random(seed)
        grouped: dict[int, dict[str, list[int]]] = {
            outcome: defaultdict(list) for outcome in (-1, 0, 1)
        }
        for index, record in enumerate(records):
            if record.outcome_value is None:
                raise ValueError("value sampler accepts only known outcomes")
            grouped[record.outcome_value][record.game_id].append(index)
        if any(not games for games in grouped.values()):
            raise ValueError("value sampler requires win, draw, and loss rows")
        self._grouped = grouped
        self._games = {outcome: tuple(sorted(games)) for outcome, games in grouped.items()}

    def sample_indices(self, size: int) -> tuple[int, ...]:
        if size <= 0:
            raise ValueError("value sample size must be positive")
        outcomes = (-1, 0, 1)
        offset = self._rng.randrange(len(outcomes))
        selected = []
        for row in range(size):
            outcome = outcomes[(row + offset) % len(outcomes)]
            game = self._rng.choice(self._games[outcome])
            selected.append(self._rng.choice(self._grouped[outcome][game]))
        self._rng.shuffle(selected)
        return tuple(selected)


class JointPolicyValueLearner:
    def __init__(self, network, *, learning_rate: float, weight_decay: float) -> None:
        self.network = network
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        self._loss_and_grad = nn.value_and_grad(network, self._loss)

    def _loss(
        self,
        policy_inputs: mx.array,
        policy_targets: mx.array,
        legal_masks: mx.array,
        value_inputs: mx.array,
        value_targets: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        policy_logits, _ = self.network(policy_inputs)
        _, value_logits = self.network(value_inputs)
        policy_logits = mx.where(legal_masks, policy_logits, mx.array(-1e9))
        policy_loss = nn.losses.cross_entropy(policy_logits, policy_targets, reduction="mean")
        value_loss = nn.losses.cross_entropy(value_logits, value_targets, reduction="mean")
        return policy_loss + value_loss, policy_loss, value_loss

    def train_step(
        self,
        policy_inputs: mx.array,
        policy_targets: mx.array,
        legal_masks: mx.array,
        value_inputs: mx.array,
        value_targets: mx.array,
    ) -> tuple[float, float, float, float]:
        (total, policy, value), gradients = self._loss_and_grad(
            policy_inputs,
            policy_targets,
            legal_masks,
            value_inputs,
            value_targets,
        )
        gradients, norm = optim.clip_grad_norm(gradients, 5.0)
        finite = mx.all(
            mx.stack([mx.all(mx.isfinite(array)) for _, array in tree_flatten(gradients)])
        )
        mx.eval(total, policy, value, norm, finite, gradients)
        if not bool(finite.item()) or not all(
            math.isfinite(float(item.item())) for item in (total, policy, value, norm)
        ):
            raise RuntimeError("joint transfer produced non-finite loss or gradients")
        self.optimizer.update(self.network, gradients)
        mx.eval(self.network.parameters(), self.optimizer.state)
        return tuple(float(item.item()) for item in (total, policy, value, norm))  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parameter_hash(network, *, excluded_prefixes: tuple[str, ...] = ()) -> str:
    digest = hashlib.sha256()
    for name, array in tree_flatten(network.parameters()):
        if name.startswith(excluded_prefixes):
            continue
        digest.update(name.encode())
        digest.update(json.dumps(array.tolist(), separators=(",", ":")).encode())
    return digest.hexdigest()


def _known_records(shard: ReplayShard) -> tuple[ReplayRecord, ...]:
    return tuple(record for record in shard.records if record.outcome_value is not None)


def _audit_perspective(records: tuple[ReplayRecord, ...]) -> dict[str, int]:
    by_game: dict[str, list[ReplayRecord]] = defaultdict(list)
    for record in records:
        by_game[record.game_id].append(record)
    decisive = draws = unknown = 0
    for game_records in by_game.values():
        outcomes = {record.outcome_value for record in game_records}
        if outcomes == {None}:
            unknown += 1
            continue
        if outcomes == {0}:
            draws += 1
            continue
        if None in outcomes or 0 in outcomes:
            raise ValueError("game mixes terminal and unknown/draw targets")
        winner_signs = {
            int(record.outcome_value) * (1 if record.side_to_move is Side.WHITE else -1)
            for record in game_records
        }
        if len(winner_signs) != 1 or not outcomes <= {-1, 1}:
            raise ValueError("decisive outcome perspective does not alternate by side")
        decisive += 1
    return {"games": len(by_game), "decisive": decisive, "draw": draws, "unknown": unknown}


def _value_logits(network, batch: PreparedTrainingBatch) -> mx.array:
    logits = network._value_logits(network._trunk(batch.inputs))
    mx.eval(logits)
    return logits


def _value_quality(logits: mx.array, outcomes: tuple[int | None, ...]) -> dict[str, object]:
    if any(outcome is None for outcome in outcomes):
        raise ValueError("value quality accepts only known outcomes")
    rows = logits.tolist()
    concrete = tuple(int(outcome) for outcome in outcomes if outcome is not None)
    base = _wdl_quality(rows, concrete)
    class_index = {1: 0, 0: 1, -1: 2}
    probabilities = []
    for row in rows:
        maximum = max(row)
        weights = [math.exp(value - maximum) for value in row]
        total = sum(weights)
        probabilities.append(tuple(value / total for value in weights))
    class_ce = {
        str(outcome): mean(
            -math.log(max(probabilities[index][class_index[outcome]], 1e-300))
            for index, target in enumerate(concrete)
            if target == outcome
        )
        for outcome in (-1, 0, 1)
    }
    expected = [row[0] - row[2] for row in probabilities]
    means = {
        str(outcome): mean(
            value for value, target in zip(expected, concrete, strict=True) if target == outcome
        )
        for outcome in (-1, 0, 1)
    }
    return {
        **base,
        "macro_cross_entropy": mean(class_ce.values()),
        "class_cross_entropy": class_ce,
        "accuracy": mean(
            max(range(3), key=probabilities[index].__getitem__) == class_index[target]
            for index, target in enumerate(concrete)
        ),
        "expected_value_by_outcome": means,
        "loss_draw_margin": means["0"] - means["-1"],
        "win_draw_margin": means["1"] - means["0"],
    }


def _value_gate_reasons(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    train: dict[str, object] | None = None,
    enforce_ece: bool = True,
) -> tuple[str, ...]:
    reasons = []
    if float(baseline["cross_entropy"]) - float(candidate["cross_entropy"]) < 0.10:
        reasons.append("validation micro WDL CE improvement is below 0.10")
    if float(baseline["macro_cross_entropy"]) - float(candidate["macro_cross_entropy"]) < 0.10:
        reasons.append("validation macro WDL CE improvement is below 0.10")
    if float(baseline["brier"]) - float(candidate["brier"]) < 0.03:
        reasons.append("validation WDL Brier improvement is below 0.03")
    if float(candidate["expected_score_pearson"]) < 0.20:
        reasons.append("validation expected-score Pearson is below 0.20")
    if min(float(candidate["loss_draw_margin"]), float(candidate["win_draw_margin"])) < 0.03:
        reasons.append("validation outcome means lack ordered 0.03 margins")
    if enforce_ece and float(candidate["ece_10"]) > 0.12:
        reasons.append("validation WDL ECE-10 exceeds 0.12")
    if train is not None and (
        float(candidate["macro_cross_entropy"]) - float(train["macro_cross_entropy"]) > 0.15
    ):
        reasons.append("validation macro WDL CE trails train by over 0.15")
    return tuple(reasons)


def _policy_gate_reasons(
    baseline: dict[str, dict[str, float]], candidate: dict[str, dict[str, float]]
) -> tuple[str, ...]:
    reasons = []
    before = baseline["validation"]
    after = candidate["validation"]
    if before["cross_entropy"] - after["cross_entropy"] < 0.05:
        reasons.append("validation policy CE improvement is below 0.05")
    if after["top_action_agreement"] - before["top_action_agreement"] < 0.02:
        reasons.append("validation policy top-action gain is below two points")
    if after["top_action_agreement"] < candidate["train"]["top_action_agreement"] - 0.15:
        reasons.append("validation policy agreement trails train by over 15 points")
    return tuple(reasons)


def _ranks(values: Sequence[float]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and math.isclose(
            values[order[start]], values[order[end]], abs_tol=1e-12
        ):
            end += 1
        rank = (start + end - 1) / 2.0
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return tuple(ranks)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = _ranks(left)
    y = _ranks(right)
    mean_x, mean_y = mean(x), mean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def _continuation_ranking(
    baseline,
    candidate,
    records: tuple[ReplayRecord, ...],
    *,
    rules: PythonChessRules,
    depth: int,
) -> dict[str, object]:
    encoder = BoardEncoder(rules)
    oracle = DeterministicTacticalOracle(rules=rules, config=TacticalOracleConfig(depth=depth))
    baseline_correlations = []
    candidate_correlations = []
    baseline_top = []
    candidate_top = []
    rows = []
    for record in records:
        moves = tuple(rules.legal_moves(record.state))
        children = tuple(rules.apply(record.state, move) for move in moves)
        positions = tuple(encoder.encode(child) for child in children)
        shape = positions[0].shape
        inputs = mx.array([position.values for position in positions], dtype=mx.float32).reshape(
            (len(positions), *shape)
        )
        baseline_logits = baseline._value_logits(baseline._trunk(inputs))
        candidate_logits = candidate._value_logits(candidate._trunk(inputs))
        mx.eval(baseline_logits, candidate_logits)

        def expected(rows: list[list[float]]) -> tuple[float, ...]:
            values = []
            for row in rows:
                maximum = max(row)
                weights = [math.exp(value - maximum) for value in row]
                total = sum(weights)
                values.append(-(weights[0] - weights[2]) / total)
            return tuple(values)

        baseline_q = expected(baseline_logits.tolist())
        candidate_q = expected(candidate_logits.tolist())
        verified_q = tuple(-oracle.value(child) for child in children)
        baseline_rho = _spearman(baseline_q, verified_q)
        candidate_rho = _spearman(candidate_q, verified_q)
        best = max(verified_q)
        verified_best = {index for index, value in enumerate(verified_q) if value == best}
        baseline_hit = max(range(len(moves)), key=baseline_q.__getitem__) in verified_best
        candidate_hit = max(range(len(moves)), key=candidate_q.__getitem__) in verified_best
        baseline_correlations.append(baseline_rho)
        candidate_correlations.append(candidate_rho)
        baseline_top.append(baseline_hit)
        candidate_top.append(candidate_hit)
        rows.append(
            {
                "identity": f"{record.game_id}:{record.game_index}:{record.ply}",
                "legal_actions": len(moves),
                "baseline_spearman": baseline_rho,
                "candidate_spearman": candidate_rho,
                "baseline_verified_top": baseline_hit,
                "candidate_verified_top": candidate_hit,
            }
        )
    result = {
        "positions": len(records),
        "baseline_mean_spearman": mean(baseline_correlations),
        "candidate_mean_spearman": mean(candidate_correlations),
        "spearman_improvement": mean(candidate_correlations) - mean(baseline_correlations),
        "baseline_verified_top_agreement": mean(baseline_top),
        "candidate_verified_top_agreement": mean(candidate_top),
        "rows": rows,
    }
    reasons = []
    if float(result["spearman_improvement"]) < 0.05:
        reasons.append("continuation mean Spearman improvement is below 0.05")
    if float(result["candidate_mean_spearman"]) <= 0:
        reasons.append("candidate continuation mean Spearman is not positive")
    if float(result["candidate_verified_top_agreement"]) < float(
        result["baseline_verified_top_agreement"]
    ):
        reasons.append("candidate continuation verified-top agreement regressed")
    result["passed"] = not reasons
    result["reasons"] = reasons
    return result


def _quality(network, policy, value: PreparedTrainingBatch, outcomes) -> dict[str, object]:
    policy_logits, _ = network(policy.inputs)
    value_logits = _value_logits(network, value)
    mx.eval(policy_logits)
    return {
        "policy": _policy_quality(policy_logits, policy.targets, policy.legal_masks),
        "value": _value_quality(value_logits, outcomes),
    }


def _transfer_config(config: JointPolicyValueTransferConfig) -> FullGumbelTransferConfig:
    return FullGumbelTransferConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        target_result=config.target_result,
        train_shard=config.train_shard,
        validation_shard=config.validation_shard,
        telemetry_path=config.telemetry_path,
        seed=config.seed,
        arena_seed=config.arena_seed,
    )


def run_joint_policy_value_transfer(config: JointPolicyValueTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"joint transfer output exists: {config.output_dir}")
    target = json.loads(config.target_result.read_text(encoding="utf-8"))
    if not target.get("passed") or not target.get("learner_transfer_authorized"):
        raise ValueError("joint transfer requires qualified Full Gumbel targets")
    if _sha256(config.model_path) != config.expected_model_sha256:
        raise ValueError("joint transfer baseline checksum mismatch")
    if target.get("config", {}).get("model_sha256") != config.expected_model_sha256:
        raise ValueError("qualified policy targets use a different baseline")
    if (
        len(target.get("rows", {}).get("train", ())) != 384
        or len(target.get("rows", {}).get("validation", ())) != 192
    ):
        raise ValueError("qualified policy target cardinality changed")
    rules = PythonChessRules()
    train_shard = read_shard(config.train_shard, rules=rules)
    validation_shard = read_shard(config.validation_shard, rules=rules)
    if min(train_shard.header.target_schema, validation_shard.header.target_schema) < 10:
        raise ValueError("joint transfer requires corrected max-ply target schema")
    train_games = {record.game_id for record in train_shard.records}
    validation_games = {record.game_id for record in validation_shard.records}
    if train_games & validation_games:
        raise ValueError("joint transfer train and validation games overlap")
    perspective = {
        "train": _audit_perspective(train_shard.records),
        "validation": _audit_perspective(validation_shard.records),
    }
    train_policy = _prepare(train_shard.records, target["rows"]["train"], rules=rules)
    validation_policy = _prepare(
        validation_shard.records, target["rows"]["validation"], rules=rules
    )
    train_value_records = _known_records(train_shard)
    validation_value_records = _known_records(validation_shard)
    train_value = MLXLearner.prepare_batch(build_training_batch(train_value_records, rules=rules))
    validation_value = MLXLearner.prepare_batch(
        build_training_batch(validation_value_records, rules=rules)
    )
    train_outcomes = tuple(record.outcome_value for record in train_value_records)
    validation_outcomes = tuple(record.outcome_value for record in validation_value_records)
    ranking_records = select_stratified_records(
        validation_shard.records,
        rules=rules,
        count=config.ranking_positions,
        seed=config.seed,
    )
    config.output_dir.mkdir(parents=True)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="KRITIK value-head learnability control",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.warmup_steps + config.joint_steps,
        pilot_steps_completed=0,
        pilot_validation_interval_steps=config.validation_interval,
        pilot_early_stopping_patience=config.early_stopping_patience,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()

    baseline = _network()
    baseline.load_weights(str(config.model_path))
    baseline_quality = {
        "train": _quality(baseline, train_policy, train_value, train_outcomes),
        "validation": _quality(baseline, validation_policy, validation_value, validation_outcomes),
    }
    warmup = _network()
    warmup.load_weights(str(config.model_path))
    frozen_before = _parameter_hash(
        warmup, excluded_prefixes=("value_conv.", "value_hidden.", "value_output.")
    )
    _freeze_to_value_head(warmup)
    warmup_learner = MLXLearner(
        warmup,
        config=LearnerConfig(
            learning_rate=config.warmup_learning_rate,
            weight_decay=0.0,
            policy_weight=0.0,
            value_weight=1.0,
        ),
    )
    value_sampler = OutcomeGameBalancedSampler(train_value_records, seed=config.seed)
    warmup_curve = []
    warmup_best = None
    warmup_best_ce = math.inf
    warmup_stale = 0
    warmup_stop = "maximum_steps"
    maximum_gradient_norm = 0.0
    for step in range(1, config.warmup_steps + 1):
        metrics = warmup_learner.train_step(
            train_value.select(value_sampler.sample_indices(config.batch_size))
        )
        maximum_gradient_norm = max(maximum_gradient_norm, metrics.unclipped_gradient_norm)
        if step % config.validation_interval:
            continue
        quality = _value_quality(_value_logits(warmup, validation_value), validation_outcomes)
        warmup_curve.append({"step": step, "quality": quality})
        macro = float(quality["macro_cross_entropy"])
        if macro < warmup_best_ce:
            warmup_best_ce = macro
            warmup_best = warmup_learner.snapshot()
            warmup_stale = 0
        else:
            warmup_stale += 1
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode_detail=(
                f"KRITIK value warmup · {step}/{config.warmup_steps} · macro CE {macro:.4f}"
            ),
            pilot_steps_completed=step,
            training_step=step,
            value_loss=float(quality["cross_entropy"]),
            pilot_last_validation_step=step,
            pilot_last_validation_loss=macro,
        )
        store.write_atomic(snapshot)
        if warmup_stale >= config.early_stopping_patience:
            warmup_stop = "validation_early_stopping"
            break
    if warmup_best is None:
        raise RuntimeError("value warmup produced no checkpoint")
    warmup_learner.restore(warmup_best)
    warmup_quality = {
        "train": _value_quality(_value_logits(warmup, train_value), train_outcomes),
        "validation": _value_quality(_value_logits(warmup, validation_value), validation_outcomes),
    }
    warmup_reasons = list(
        _value_gate_reasons(
            baseline_quality["validation"]["value"],  # type: ignore[arg-type]
            warmup_quality["validation"],
            enforce_ece=False,
        )
    )
    frozen_after = _parameter_hash(
        warmup, excluded_prefixes=("value_conv.", "value_hidden.", "value_output.")
    )
    if frozen_before != frozen_after:
        warmup_reasons.append("value warmup changed a frozen non-value parameter")
    warmup_path = config.output_dir / "warmup" / "model.safetensors"
    warmup_path.parent.mkdir()
    warmup.save_weights(str(warmup_path))
    warmup_result = {
        "passed": not warmup_reasons,
        "reasons": warmup_reasons,
        "stop_reason": warmup_stop,
        "selected_step": warmup_best.step,
        "maximum_gradient_norm": maximum_gradient_norm,
        "quality": warmup_quality,
        "curve": warmup_curve,
        "model_path": str(warmup_path),
        "model_sha256": _sha256(warmup_path),
        "frozen_non_value_hash_before": frozen_before,
        "frozen_non_value_hash_after": frozen_after,
    }

    joint_result = continuation = tactical = candidate = arena = arena_control = arena_gate = None
    passed = False
    if not warmup_reasons or not config.require_head_warmup_gate:
        transfer = warmup
        joint_start = "qualified_value_head_warmup"
        if warmup_reasons:
            transfer = _network()
            transfer.load_weights(str(config.model_path))
            joint_start = "release_baseline_shared_representation_audit"
        transfer.unfreeze()
        joint = JointPolicyValueLearner(
            transfer,
            learning_rate=config.joint_learning_rate,
            weight_decay=config.weight_decay,
        )
        policy_rng = random.Random(config.seed)
        value_sampler = OutcomeGameBalancedSampler(train_value_records, seed=config.seed)
        joint_curve = []
        eligible = []
        stale = 0
        best_macro = math.inf
        joint_stop = "maximum_steps"
        for step in range(1, config.joint_steps + 1):
            policy_indices = _select_indices(train_policy.records, config.batch_size, policy_rng)
            value_indices = value_sampler.sample_indices(config.batch_size)
            total, policy_loss, value_loss, norm = joint.train_step(
                _take(train_policy.inputs, policy_indices),
                _take(train_policy.targets, policy_indices),
                _take(train_policy.legal_masks, policy_indices),
                mx.take(train_value.inputs, mx.array(value_indices, dtype=mx.int32), axis=0),
                mx.take(
                    train_value.wdl_targets,
                    mx.array(value_indices, dtype=mx.int32),
                    axis=0,
                ),
            )
            maximum_gradient_norm = max(maximum_gradient_norm, norm)
            if step % config.validation_interval:
                continue
            quality = {
                "train": _quality(transfer, train_policy, train_value, train_outcomes),
                "validation": _quality(
                    transfer, validation_policy, validation_value, validation_outcomes
                ),
            }
            reasons = (
                *_value_gate_reasons(
                    baseline_quality["validation"]["value"],  # type: ignore[arg-type]
                    quality["validation"]["value"],  # type: ignore[arg-type]
                    train=quality["train"]["value"],  # type: ignore[arg-type]
                ),
                *_policy_gate_reasons(
                    {
                        "train": baseline_quality["train"]["policy"],
                        "validation": baseline_quality["validation"]["policy"],
                    },  # type: ignore[arg-type]
                    {
                        "train": quality["train"]["policy"],
                        "validation": quality["validation"]["policy"],
                    },  # type: ignore[arg-type]
                ),
            )
            row = {
                "step": step,
                "batch_total_loss": total,
                "batch_policy_loss": policy_loss,
                "batch_value_loss": value_loss,
                "quality": quality,
                "numeric_gate_reasons": reasons,
            }
            joint_curve.append(row)
            macro = float(quality["validation"]["value"]["macro_cross_entropy"])  # type: ignore[index]
            if not reasons:
                eligible.append((macro, step, _snapshot(transfer)))
            if macro < best_macro:
                best_macro = macro
                stale = 0
            else:
                stale += 1
            snapshot = replace(
                snapshot,
                updated_at=datetime.now(UTC).isoformat(),
                mode_detail=(
                    f"KRITIK joint transfer · {step}/{config.joint_steps} · macro CE {macro:.4f}"
                ),
                pilot_steps_completed=config.warmup_steps + step,
                training_step=config.warmup_steps + step,
                policy_loss=policy_loss,
                value_loss=value_loss,
                total_loss=total,
                pilot_last_validation_step=step,
                pilot_last_validation_loss=macro,
            )
            store.write_atomic(snapshot)
            if stale >= config.early_stopping_patience:
                joint_stop = "validation_early_stopping"
                break
        if eligible:
            selected_macro, selected_step, selected_weights = min(
                eligible, key=lambda item: (item[0], item[1])
            )
            transfer.load_weights(list(selected_weights))
            selected_quality = {
                "train": _quality(transfer, train_policy, train_value, train_outcomes),
                "validation": _quality(
                    transfer, validation_policy, validation_value, validation_outcomes
                ),
            }
            candidate_dir = config.output_dir / "candidate"
            candidate_dir.mkdir()
            candidate_path = candidate_dir / "model.safetensors"
            temporary = candidate_dir / ".model.tmp.safetensors"
            transfer.save_weights(str(temporary))
            os.replace(temporary, candidate_path)
            continuation = _continuation_ranking(
                baseline,
                transfer,
                ranking_records,
                rules=rules,
                depth=config.ranking_depth,
            )
            transfer_config = _transfer_config(config)
            baseline_tactical = _tactical(config.model_path, config=transfer_config)
            candidate_tactical = _tactical(candidate_path, config=transfer_config)
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
            tactical_reasons = []
            if int(candidate_tactical["raw"]["solved"]) < int(  # type: ignore[index]
                baseline_tactical["raw"]["solved"]  # type: ignore[index]
            ):
                tactical_reasons.append("raw tactical solve count regressed")
            if int(candidate_tactical["budgets"][0]["solved"]) < 4:  # type: ignore[index]
                tactical_reasons.append("256 Full Gumbel tactical solve count is below four")
            if baseline_cases - candidate_cases:
                tactical_reasons.append("candidate search lost a baseline-solved tactical case")
            tactical = {
                "baseline": baseline_tactical,
                "candidate": candidate_tactical,
                "passed": not tactical_reasons,
                "reasons": tactical_reasons,
            }
            candidate = {
                "path": str(candidate_path),
                "sha256": _sha256(candidate_path),
                "selected_step": selected_step,
                "selected_macro_wdl_ce": selected_macro,
            }
            prearena_reasons = [*continuation["reasons"], *tactical_reasons]
            if not prearena_reasons:
                snapshot = replace(
                    snapshot,
                    updated_at=datetime.now(UTC).isoformat(),
                    mode=RunMode.EVALUATION,
                    mode_detail="KRITIK qualified joint checkpoint · fresh paired arena",
                )
                store.write_atomic(snapshot)
                arena, arena_control, inherited_gate = _arena(
                    config.model_path, candidate_path, config=transfer_config
                )
                strict_reasons = list(inherited_gate["reasons"])
                if float(arena["score_interval"]["low"]) <= 0.50:  # type: ignore[index]
                    strict_reasons.append("paired expected-score lower bound is not positive")
                if float(arena["decisive_score"]) < 0.50:
                    strict_reasons.append("decisive score regressed")
                arena_gate = {"passed": not strict_reasons, "reasons": strict_reasons}
                passed = not strict_reasons
            joint_result = {
                "joint_start": joint_start,
                "passed_numeric_gate": True,
                "stop_reason": joint_stop,
                "selected_step": selected_step,
                "selected_quality": selected_quality,
                "curve": joint_curve,
                "maximum_gradient_norm": maximum_gradient_norm,
                "prearena_reasons": prearena_reasons,
            }
        else:
            joint_result = {
                "joint_start": joint_start,
                "passed_numeric_gate": False,
                "stop_reason": joint_stop,
                "curve": joint_curve,
                "maximum_gradient_norm": maximum_gradient_norm,
                "prearena_reasons": ["no checkpoint passed frozen value and policy gates"],
            }

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
            "provenance": {
                "model_sha256": _sha256(config.model_path),
                "target_result_sha256": _sha256(config.target_result),
                "target_schema": train_shard.header.target_schema,
                "train_validation_games_disjoint": True,
                "train_value_rows": len(train_value_records),
                "validation_value_rows": len(validation_value_records),
                "train_unknown_rows_excluded": len(train_shard.records) - len(train_value_records),
                "validation_unknown_rows_excluded": len(validation_shard.records)
                - len(validation_value_records),
                "perspective": perspective,
            },
            "baseline_quality": baseline_quality,
            "warmup": warmup_result,
            "joint": joint_result,
            "continuation_ranking": continuation,
            "tactical": tactical,
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
    detail = (
        "KRITIK joint transfer passed · continuous learner may be designed"
        if passed
        else (
            "KRITIK value warmup failed · representation audit required"
            if warmup_reasons and config.require_head_warmup_gate
            else "KRITIK joint transfer failed · continuous learner blocked"
        )
    )
    stop_reasons = (
        warmup_reasons
        if warmup_reasons and config.require_head_warmup_gate
        else (joint_result or {}).get("prearena_reasons", ())
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=detail,
        pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
        pilot_stop_reason=(
            "all_joint_transfer_gates_passed" if passed else "frozen_joint_transfer_gate"
        ),
        pilot_stop_detail=detail,
        pilot_reasons=tuple(stop_reasons),  # type: ignore[arg-type]
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
    parser.add_argument(
        "--joint-representation-audit",
        action="store_true",
        help="run the preregistered all-network joint arm even when head-only warmup fails",
    )
    arguments = parser.parse_args(argv)
    print(
        run_joint_policy_value_transfer(
            JointPolicyValueTransferConfig(
                output_dir=arguments.output_dir,
                model_path=arguments.model,
                target_result=arguments.target_result,
                train_shard=arguments.train_shard,
                validation_shard=arguments.validation_shard,
                telemetry_path=arguments.telemetry,
                require_head_warmup_gate=not arguments.joint_representation_audit,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
