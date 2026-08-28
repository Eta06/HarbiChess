"""Frozen policy-head transfer from qualified Full Gumbel soft targets."""

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
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import POLICY_SIZE, legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.arena import _openings
from harbichess.evaluation.system_teacher_qualification import (
    QualificationGame,
    _play_game,
    summarize_control,
    summarize_games,
)
from harbichess.evaluation.teacher_qualification import _atomic_json
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.diagnostics import run_tactical_sweep
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS


@dataclass(frozen=True, slots=True)
class FullGumbelTransferConfig:
    output_dir: Path
    model_path: Path
    target_result: Path
    train_shard: Path
    validation_shard: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    learning_rate: float = 2e-4
    batch_size: int = 64
    maximum_steps: int = 240
    validation_interval: int = 20
    early_stopping_patience: int = 4
    seed: int = 2026082883
    arena_seed: int = 2026082893
    arena_opening_pairs: int = 32
    arena_opening_plies: int = 8
    arena_max_plies: int = 256
    arena_simulations: int = 128
    search_workers: int = 24
    fixed_inference_batch_size: int = 12
    inference_wait_seconds: float = 0.00025
    bootstrap_samples: int = 50_000

    def __post_init__(self) -> None:
        counts = (
            self.batch_size,
            self.maximum_steps,
            self.validation_interval,
            self.early_stopping_patience,
            self.seed,
            self.arena_seed,
            self.arena_opening_pairs,
            self.arena_max_plies,
            self.arena_simulations,
            self.search_workers,
            self.fixed_inference_batch_size,
            self.bootstrap_samples,
        )
        if any(value <= 0 for value in counts) or self.arena_opening_plies < 0:
            raise ValueError("Full Gumbel transfer counts must be positive")
        if self.maximum_steps % self.validation_interval:
            raise ValueError("maximum steps must align with validation interval")
        if self.learning_rate <= 0 or self.inference_wait_seconds < 0:
            raise ValueError("Full Gumbel transfer rates are invalid")


@dataclass(frozen=True, slots=True)
class PreparedTransfer:
    records: tuple[ReplayRecord, ...]
    inputs: mx.array
    targets: mx.array
    legal_masks: mx.array
    wdl_targets: tuple[int | None, ...]


class PolicyHead(nn.Module):
    def __init__(self, network: HarbiChessNetwork) -> None:
        super().__init__()
        self.policy_conv = network.policy_conv
        self.policy_linear = network.policy_linear

    def __call__(self, trunk: mx.array) -> mx.array:
        features = nn.relu(self.policy_conv(trunk)).reshape(trunk.shape[0], -1)
        return self.policy_linear(features)


class PolicyHeadLearner:
    def __init__(self, head: PolicyHead, *, learning_rate: float) -> None:
        self.head = head
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self._loss_and_grad = nn.value_and_grad(self.head, self._loss)

    def _loss(
        self,
        trunk: mx.array,
        targets: mx.array,
        legal_masks: mx.array,
    ) -> mx.array:
        logits = mx.where(legal_masks, self.head(trunk), mx.array(-1e9))
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def train_step(
        self,
        trunk: mx.array,
        targets: mx.array,
        legal_masks: mx.array,
    ) -> tuple[float, float]:
        loss, gradients = self._loss_and_grad(trunk, targets, legal_masks)
        gradients, norm = optim.clip_grad_norm(gradients, 5.0)
        finite = mx.all(
            mx.stack([mx.all(mx.isfinite(value)) for _, value in tree_flatten(gradients)])
        )
        mx.eval(loss, norm, finite, gradients)
        if not bool(finite.item()) or not math.isfinite(float(loss.item())):
            raise RuntimeError("policy-head transfer produced non-finite gradients")
        self.optimizer.update(self.head, gradients)
        mx.eval(self.head.parameters(), self.optimizer.state)
        return float(loss.item()), float(norm.item())


def _network() -> HarbiChessNetwork:
    return HarbiChessNetwork(
        NetworkConfig(
            trunk_channels=16,
            residual_blocks=2,
            policy_channels=4,
            value_channels=2,
            value_hidden=32,
        )
    )


def _identity(record: ReplayRecord) -> str:
    return f"{record.game_id}:{record.game_index}:{record.ply}"


def _prepare(
    records: tuple[ReplayRecord, ...],
    target_rows: Sequence[Mapping[str, object]],
    *,
    rules: PythonChessRules,
) -> PreparedTransfer:
    by_identity = {_identity(record): record for record in records}
    encoder = BoardEncoder(rules)
    selected = []
    positions = []
    targets = []
    masks = []
    outcomes = []
    for row in target_rows:
        record = by_identity[str(row["identity"])]
        board = rules.board(record.state)
        selected.append(record)
        positions.append(encoder.encode_state(record.state, board))
        dense = [0.0] * POLICY_SIZE
        for action, probability in row["action_target"]:  # type: ignore[misc]
            dense[int(action)] = float(probability)
        targets.append(dense)
        legal = [False] * POLICY_SIZE
        for action in legal_action_indices(board):
            legal[action] = True
        masks.append(legal)
        outcomes.append(record.outcome_value)
    shape = positions[0].shape
    inputs = mx.array([position.values for position in positions], dtype=mx.float32)
    inputs = inputs.reshape((len(positions), *shape))
    prepared = PreparedTransfer(
        tuple(selected),
        inputs,
        mx.array(targets, dtype=mx.float32),
        mx.array(masks, dtype=mx.bool_),
        tuple(outcomes),
    )
    mx.eval(prepared.inputs, prepared.targets, prepared.legal_masks)
    return prepared


def _policy_quality(logits: mx.array, targets: mx.array, legal_masks: mx.array) -> dict[str, float]:
    masked = mx.where(legal_masks, logits, mx.array(-1e9))
    log_probs = masked - mx.logsumexp(masked, axis=1, keepdims=True)
    cross_entropy = -mx.mean(mx.sum(targets * log_probs, axis=1))
    entropy = -mx.mean(
        mx.sum(mx.where(targets > 0, targets * mx.log(targets), mx.array(0.0)), axis=1)
    )
    agreement = mx.mean(
        (mx.argmax(masked, axis=1) == mx.argmax(targets, axis=1)).astype(mx.float32)
    )
    mx.eval(cross_entropy, entropy, agreement)
    ce = float(cross_entropy.item())
    return {
        "cross_entropy": ce,
        "target_entropy": float(entropy.item()),
        "teacher_kl": ce - float(entropy.item()),
        "top_action_agreement": float(agreement.item()),
    }


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    maximum = max(values)
    weights = tuple(math.exp(value - maximum) for value in values)
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = tuple(value - mean_x for value in xs)
    centered_y = tuple(value - mean_y for value in ys)
    denominator = math.sqrt(
        sum(value * value for value in centered_x) * sum(value * value for value in centered_y)
    )
    return (
        sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / denominator
        if denominator
        else 0.0
    )


def _wdl_quality(
    logits: Sequence[Sequence[float]], outcomes: Sequence[int | None]
) -> dict[str, float | int]:
    rows = [
        (_softmax(tuple(map(float, row))), outcome)
        for row, outcome in zip(logits, outcomes, strict=True)
        if outcome is not None
    ]
    if not rows:
        raise ValueError("WDL quality requires known outcomes")
    classes = {-1: 2, 0: 1, 1: 0}
    ce = -sum(
        math.log(max(probabilities[classes[outcome]], 1e-300)) for probabilities, outcome in rows
    ) / len(rows)  # type: ignore[index]
    brier = sum(
        sum(
            (probability - float(index == classes[outcome])) ** 2
            for index, probability in enumerate(probabilities)
        )
        for probabilities, outcome in rows
    ) / len(rows)
    expected = [probabilities[0] - probabilities[2] for probabilities, _ in rows]
    observed = [float(outcome) for _, outcome in rows]
    bins: list[list[tuple[float, float]]] = [[] for _ in range(10)]
    for probabilities, outcome in rows:
        confidence = max(probabilities)
        prediction = max(range(3), key=lambda index: probabilities[index])
        bins[min(9, int(confidence * 10))].append(
            (confidence, float(prediction == classes[outcome]))
        )
    ece = sum(
        len(bucket)
        / len(rows)
        * abs(
            sum(confidence for confidence, _ in bucket) / len(bucket)
            - sum(accuracy for _, accuracy in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )
    return {
        "known_positions": len(rows),
        "cross_entropy": ce,
        "brier": brier,
        "expected_score_pearson": _pearson(expected, observed),
        "ece_10": ece,
    }


def _parameter_hash(network: HarbiChessNetwork, *, policy: bool) -> str:
    digest = hashlib.sha256()
    for name, array in tree_flatten(network.parameters()):
        is_policy = name.startswith("policy_conv.") or name.startswith("policy_linear.")
        if is_policy != policy:
            continue
        digest.update(name.encode())
        digest.update(
            json.dumps(array.tolist(), separators=(",", ":"), allow_nan=False).encode()
        )
    return digest.hexdigest()


def _snapshot(head: PolicyHead) -> tuple[tuple[str, mx.array], ...]:
    return tuple((name, mx.array(value)) for name, value in tree_flatten(head.parameters()))


def _select_indices(
    records: tuple[ReplayRecord, ...], size: int, rng: random.Random
) -> tuple[int, ...]:
    by_game: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_game[record.game_id].append(index)
    games = tuple(sorted(by_game))
    chosen_games = rng.sample(games, len(games)) + rng.choices(games, k=max(0, size - len(games)))
    return tuple(rng.choice(by_game[game]) for game in chosen_games[:size])


def _take(array: mx.array, indices: tuple[int, ...]) -> mx.array:
    return mx.take(array, mx.array(indices, dtype=mx.int32), axis=0)


def _evaluator(network: HarbiChessNetwork, workers: int, fixed_batch: int, wait: float):
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network, fixed_batch_size=fixed_batch),
        max_batch_size=min(workers, fixed_batch),
        max_wait_seconds=wait,
    )
    return batcher, NeuralPositionEvaluator(batcher)


def _tactical(
    model_path: Path,
    *,
    config: FullGumbelTransferConfig,
) -> dict[str, object]:
    network = _network()
    network.load_weights(str(model_path))
    rules = PythonChessRules()
    batcher, evaluator = _evaluator(
        network,
        config.search_workers,
        config.fixed_inference_batch_size,
        config.inference_wait_seconds,
    )
    try:
        return run_tactical_sweep(
            evaluator,
            rules=rules,
            budgets=(256,),
            workers=8,
            seed=config.seed,
            search_kind="full-gumbel",
            max_considered_actions=16,
            gumbel_scale=0.0,
        )
    finally:
        batcher.close()


def _arena(
    baseline_path: Path,
    candidate_path: Path,
    *,
    config: FullGumbelTransferConfig,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    rules = PythonChessRules()
    baseline = _network()
    baseline.load_weights(str(baseline_path))
    candidate = _network()
    candidate.load_weights(str(candidate_path))
    baseline_batcher, baseline_evaluator = _evaluator(
        baseline,
        config.search_workers,
        config.fixed_inference_batch_size,
        config.inference_wait_seconds,
    )
    candidate_batcher, candidate_evaluator = _evaluator(
        candidate,
        config.search_workers,
        config.fixed_inference_batch_size,
        config.inference_wait_seconds,
    )
    baseline_search = FullGumbelMCTS(
        baseline_evaluator,
        rules=rules,
        config=FullGumbelConfig(simulations=config.arena_simulations),
    )
    candidate_search = FullGumbelMCTS(
        candidate_evaluator,
        rules=rules,
        config=FullGumbelConfig(simulations=config.arena_simulations),
    )
    openings = _openings(
        rules,
        count=config.arena_opening_pairs,
        plies=config.arena_opening_plies,
        seed=config.arena_seed,
    )
    tasks = [
        (pair, side, state, moves)
        for pair, (state, moves) in enumerate(openings)
        for side in ("white", "black")
    ]

    def play(task) -> QualificationGame:
        pair, side, state, moves = task
        from harbichess.core.state import Side

        candidate_side = Side.WHITE if side == "white" else Side.BLACK
        white = candidate_search if candidate_side is Side.WHITE else baseline_search
        black = candidate_search if candidate_side is Side.BLACK else baseline_search
        return _play_game(
            white,
            black,
            rules,
            state,
            pair_index=pair,
            candidate_side=candidate_side,
            opening_moves=moves,
            max_plies=config.arena_max_plies,
        )

    def control(task) -> QualificationGame:
        pair, _side, state, moves = task
        return _play_game(
            baseline_search,
            baseline_search,
            rules,
            state,
            pair_index=pair,
            candidate_side=None,
            opening_moves=moves,
            max_plies=config.arena_max_plies,
        )

    try:
        with ThreadPoolExecutor(max_workers=config.search_workers) as pool:
            games = tuple(pool.map(play, tasks))
        with ThreadPoolExecutor(max_workers=config.search_workers) as pool:
            control_games = tuple(pool.map(control, tasks))
    finally:
        baseline_batcher.close()
        candidate_batcher.close()
    summary = summarize_games(
        games, bootstrap_samples=config.bootstrap_samples, seed=config.arena_seed
    )
    baseline_control = summarize_control(control_games)
    interval = summary["score_interval"]
    reasons = []
    if float(summary["score_rate"]) < 0.50:
        reasons.append("candidate search score is below 50%")
    if float(interval["low"]) < 0.45:  # type: ignore[index]
        reasons.append("candidate paired lower bound is below 45%")
    if float(summary["decisive_score"]) < 0.50:
        reasons.append("candidate decisive score is below 50%")
    for metric in ("max_ply_rate", "threefold_rate"):
        if float(summary[metric]) > float(baseline_control[metric]) + 0.10:
            reasons.append(f"candidate {metric} exceeds baseline control margin")
    return summary, baseline_control, {"passed": not reasons, "reasons": reasons}


def run_full_gumbel_transfer(config: FullGumbelTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"Full Gumbel transfer output exists: {config.output_dir}")
    target = json.loads(config.target_result.read_text(encoding="utf-8"))
    if not target.get("passed") or not target.get("learner_transfer_authorized"):
        raise ValueError("transfer requires qualified Full Gumbel targets")
    rules = PythonChessRules()
    train = _prepare(
        read_shard(config.train_shard, rules=rules).records,
        target["rows"]["train"],
        rules=rules,
    )
    validation = _prepare(
        read_shard(config.validation_shard, rules=rules).records,
        target["rows"]["validation"],
        rules=rules,
    )
    if {record.game_id for record in train.records} & {
        record.game_id for record in validation.records
    }:
        raise ValueError("transfer train and validation games overlap")
    network = _network()
    network.load_weights(str(config.model_path))
    non_policy_before = _parameter_hash(network, policy=False)
    baseline_policy_hash = _parameter_hash(network, policy=True)
    train_trunk = mx.stop_gradient(network._trunk(train.inputs))
    validation_trunk = mx.stop_gradient(network._trunk(validation.inputs))
    baseline_train_logits = network.policy_linear(network._policy_features(train_trunk))
    baseline_validation_logits = network.policy_linear(network._policy_features(validation_trunk))
    baseline_wdl = network._value_logits(validation_trunk)
    mx.eval(
        train_trunk,
        validation_trunk,
        baseline_train_logits,
        baseline_validation_logits,
        baseline_wdl,
    )
    baseline_quality = {
        "train": _policy_quality(baseline_train_logits, train.targets, train.legal_masks),
        "validation": _policy_quality(
            baseline_validation_logits, validation.targets, validation.legal_masks
        ),
    }
    baseline_wdl_rows = baseline_wdl.tolist()
    baseline_wdl_quality = _wdl_quality(baseline_wdl_rows, validation.wdl_targets)
    head = PolicyHead(network)
    learner = PolicyHeadLearner(head, learning_rate=config.learning_rate)
    rng = random.Random(config.seed)
    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail=f"AKTAR policy transfer · 0/{config.maximum_steps}",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.maximum_steps,
        pilot_steps_completed=0,
    )
    store.write_atomic(snapshot)
    checkpoints = []
    best_ce = math.inf
    best_weights = None
    stale_checks = 0
    maximum_gradient_norm = 0.0
    started = time.perf_counter()
    stop_reason = "maximum_steps"
    for step in range(1, config.maximum_steps + 1):
        indices = _select_indices(train.records, config.batch_size, rng)
        _loss, gradient_norm = learner.train_step(
            _take(train_trunk, indices),
            _take(train.targets, indices),
            _take(train.legal_masks, indices),
        )
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        if step % config.validation_interval == 0:
            train_logits = head(train_trunk)
            validation_logits = head(validation_trunk)
            mx.eval(train_logits, validation_logits)
            quality = {
                "train": _policy_quality(train_logits, train.targets, train.legal_masks),
                "validation": _policy_quality(
                    validation_logits, validation.targets, validation.legal_masks
                ),
            }
            checkpoints.append({"step": step, "quality": quality})
            validation_ce = quality["validation"]["cross_entropy"]
            if validation_ce < best_ce:
                best_ce = validation_ce
                best_weights = _snapshot(head)
                stale_checks = 0
            else:
                stale_checks += 1
            snapshot = replace(
                snapshot,
                updated_at=datetime.now(UTC).isoformat(),
                mode_detail=(
                    f"AKTAR policy transfer · {step}/{config.maximum_steps} · "
                    f"val CE {validation_ce:.4f}"
                ),
                pilot_steps_completed=step,
            )
            store.write_atomic(snapshot)
            if stale_checks >= config.early_stopping_patience:
                stop_reason = "validation_early_stopping"
                break
    if best_weights is None:
        raise RuntimeError("transfer produced no validation checkpoint")
    head.load_weights(list(best_weights))
    candidate_train_logits = head(train_trunk)
    candidate_validation_logits = head(validation_trunk)
    candidate_wdl = network._value_logits(validation_trunk)
    mx.eval(candidate_train_logits, candidate_validation_logits, candidate_wdl)
    candidate_quality = {
        "train": _policy_quality(candidate_train_logits, train.targets, train.legal_masks),
        "validation": _policy_quality(
            candidate_validation_logits, validation.targets, validation.legal_masks
        ),
    }
    candidate_wdl_rows = candidate_wdl.tolist()
    candidate_wdl_quality = _wdl_quality(candidate_wdl_rows, validation.wdl_targets)
    maximum_wdl_logit_delta = max(
        abs(float(before) - float(after))
        for before_row, after_row in zip(baseline_wdl_rows, candidate_wdl_rows, strict=True)
        for before, after in zip(before_row, after_row, strict=True)
    )
    non_policy_after = _parameter_hash(network, policy=False)
    candidate_policy_hash = _parameter_hash(network, policy=True)
    checkpoint_dir = config.output_dir / "candidate"
    checkpoint_dir.mkdir(parents=True)
    candidate_path = checkpoint_dir / "model.safetensors"
    temporary = checkpoint_dir / ".model.tmp.safetensors"
    network.save_weights(str(temporary))
    os.replace(temporary, candidate_path)
    baseline_tactical = _tactical(config.model_path, config=config)
    candidate_tactical = _tactical(candidate_path, config=config)
    baseline_solved = {
        row["case"] for row in baseline_tactical["budgets"][0]["cases"] if row["solved"]
    }
    candidate_solved = {
        row["case"] for row in candidate_tactical["budgets"][0]["cases"] if row["solved"]
    }
    reasons = []
    baseline_validation = baseline_quality["validation"]
    candidate_validation = candidate_quality["validation"]
    if baseline_validation["cross_entropy"] - candidate_validation["cross_entropy"] < 0.01:
        reasons.append("validation teacher cross-entropy improvement is below 0.01")
    if baseline_validation["teacher_kl"] - candidate_validation["teacher_kl"] < 0.01:
        reasons.append("validation teacher KL improvement is below 0.01")
    if (
        candidate_validation["top_action_agreement"] - baseline_validation["top_action_agreement"]
        < 0.02
    ):
        reasons.append("validation teacher top-action gain is below two points")
    if (
        candidate_validation["top_action_agreement"]
        < candidate_quality["train"]["top_action_agreement"] - 0.15
    ):
        reasons.append("validation top-action agreement trails train by over 15 points")
    if int(candidate_tactical["raw"]["solved"]) < int(baseline_tactical["raw"]["solved"]):
        reasons.append("candidate raw tactical solve count regressed")
    if int(candidate_tactical["budgets"][0]["solved"]) < 4:
        reasons.append("candidate 256 Full Gumbel tactical solve count is below four")
    if baseline_solved - candidate_solved:
        reasons.append("candidate search lost a baseline-solved tactical case")
    if non_policy_before != non_policy_after:
        reasons.append("frozen non-policy parameter hash changed")
    if maximum_wdl_logit_delta > 1e-7:
        reasons.append("frozen WDL logits changed")
    for metric in ("cross_entropy", "brier", "expected_score_pearson", "ece_10"):
        if abs(float(candidate_wdl_quality[metric]) - float(baseline_wdl_quality[metric])) > 1e-7:
            reasons.append(f"frozen WDL {metric} changed")
    prearena_passed = not reasons
    arena_summary = arena_control = arena_gate = None
    if prearena_passed:
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.EVALUATION,
            mode_detail="AKTAR fresh paired search arena",
        )
        store.write_atomic(snapshot)
        arena_summary, arena_control, arena_gate = _arena(
            config.model_path, candidate_path, config=config
        )
        reasons.extend(arena_gate["reasons"])
    passed = not reasons and prearena_passed
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
            "target_provenance": {
                "path": str(config.target_result),
                "sha256": hashlib.sha256(config.target_result.read_bytes()).hexdigest(),
                "algorithm": target["algorithm"],
            },
            "training": {
                "checkpoints": checkpoints,
                "selected_validation_cross_entropy": best_ce,
                "steps_completed": checkpoints[-1]["step"],
                "stop_reason": stop_reason,
                "maximum_gradient_norm": maximum_gradient_norm,
            },
            "baseline_policy_quality": baseline_quality,
            "candidate_policy_quality": candidate_quality,
            "baseline_wdl_quality": baseline_wdl_quality,
            "candidate_wdl_quality": candidate_wdl_quality,
            "maximum_wdl_logit_delta": maximum_wdl_logit_delta,
            "frozen_non_policy_hash": {
                "before": non_policy_before,
                "after": non_policy_after,
                "passed": non_policy_before == non_policy_after,
            },
            "policy_hash": {
                "baseline": baseline_policy_hash,
                "candidate": candidate_policy_hash,
            },
            "tactical": {
                "baseline": baseline_tactical,
                "candidate": candidate_tactical,
            },
            "prearena_passed": prearena_passed,
            "arena": arena_summary,
            "arena_control": arena_control,
            "arena_gate": arena_gate,
            "reasons": reasons,
            "passed": passed,
            "candidate": {
                "path": str(candidate_path),
                "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                "rejected": not passed,
            },
            "continuous_learning_authorized": passed,
            "generation_authorized": False,
            "promotion_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    snapshot = replace(
        snapshot,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "AKTAR transfer passed · continuous learner implementation authorized"
            if passed
            else "AKTAR transfer failed · continuous learner blocked"
        ),
        pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
        pilot_steps_attempted=checkpoints[-1]["step"],
        pilot_steps_completed=checkpoints[-1]["step"],
        pilot_stop_reason=stop_reason,
        pilot_stop_detail="all transfer gates passed" if passed else "; ".join(reasons),
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
    arguments = parser.parse_args(argv)
    result = run_full_gumbel_transfer(
        FullGumbelTransferConfig(
            output_dir=arguments.output_dir,
            model_path=arguments.model,
            target_result=arguments.target_result,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            telemetry_path=arguments.telemetry,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
