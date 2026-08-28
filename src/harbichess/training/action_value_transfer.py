"""Run the frozen DEGER action-value representation transfer experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import chess
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from harbichess.backends.action_value_network import HarbiChessActionValueNetwork
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import POLICY_SIZE, action_to_legal_move, move_to_action
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.consensus_target import _record_index
from harbichess.evaluation.search_q_reliability import _spearman
from harbichess.evaluation.teacher_qualification import _atomic_json, _interval, _source_commit
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.search.value_oracle import ProcessTacticalOracle, TacticalOracleConfig
from harbichess.training.batch import GameBalancedSampler
from harbichess.training.learner_transfer import _tactical_metrics, _tactical_solved


@dataclass(frozen=True, slots=True)
class ActionValueTransferConfig:
    q_reliability_result: Path
    run_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    action_value_channels: int = 4
    steps: int = 480
    batch_size: int = 16
    learning_rate: float = 2e-4
    checkpoint_steps: tuple[int, ...] = (0, 60, 120, 240, 480)
    max_gradient_norm: float = 5.0
    verifier_depth: int = 4
    verifier_workers: int = 8
    bootstrap_samples: int = 2_000
    seed: int = 2026082822
    bootstrap_seed: int = 2026082823
    minimum_mse_improvement: float = 0.20
    minimum_teacher_spearman: float = 0.35
    maximum_harmful_ratio: float = 0.10
    harmful_delta: float = -0.025
    maximum_verified_regret: float = 0.10
    minimum_best_action_coverage: float = 0.80
    maximum_logit_delta: float = 1e-7
    tactical_budget: int = 64
    tactical_workers: int = 8

    def __post_init__(self) -> None:
        if (
            min(
                self.action_value_channels,
                self.steps,
                self.batch_size,
                self.verifier_depth,
                self.verifier_workers,
                self.bootstrap_samples,
                self.seed,
                self.bootstrap_seed,
                self.tactical_budget,
                self.tactical_workers,
            )
            <= 0
            or self.learning_rate <= 0
            or self.checkpoint_steps != tuple(sorted(set(self.checkpoint_steps)))
            or self.checkpoint_steps[0] != 0
            or self.checkpoint_steps[-1] != self.steps
            or self.max_gradient_norm <= 0
            or not 0 <= self.minimum_mse_improvement < 1
            or not -1 <= self.minimum_teacher_spearman <= 1
            or not 0 <= self.maximum_harmful_ratio <= 1
            or self.harmful_delta > 0
            or self.maximum_verified_regret < 0
            or not 0 <= self.minimum_best_action_coverage <= 1
            or self.maximum_logit_delta < 0
        ):
            raise ValueError("action-value transfer configuration is invalid")


@dataclass(frozen=True, slots=True)
class PreparedActionValueData:
    records: tuple[ReplayRecord, ...]
    inputs: mx.array
    trunk: mx.array
    state_values: mx.array
    targets: mx.array
    weights: mx.array
    teacher_values: tuple[dict[int, float], ...]
    legal_actions: tuple[tuple[int, ...], ...]

    def select(self, indices: tuple[int, ...]) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        rows = mx.array(indices, dtype=mx.int32)
        return (
            mx.take(self.trunk, rows, axis=0),
            mx.take(self.state_values, rows, axis=0),
            mx.take(self.targets, rows, axis=0),
            mx.take(self.weights, rows, axis=0),
        )


class ActionValueLearner:
    def __init__(
        self,
        network: HarbiChessActionValueNetwork,
        *,
        learning_rate: float,
        max_gradient_norm: float,
    ) -> None:
        self.head = network.action_value_head
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
        self.max_gradient_norm = max_gradient_norm
        self._loss_and_grad = nn.value_and_grad(self.head, self._loss)
        self.step = 0

    def _loss(
        self,
        trunk: mx.array,
        state_values: mx.array,
        targets: mx.array,
        weights: mx.array,
    ) -> mx.array:
        predictions = self.head(trunk, state_values)
        return mx.sum(mx.square(predictions - targets) * weights) / mx.maximum(
            mx.sum(weights), mx.array(1.0)
        )

    def train_step(
        self, batch: tuple[mx.array, mx.array, mx.array, mx.array]
    ) -> tuple[float, float, float]:
        loss, gradients = self._loss_and_grad(*batch)
        gradients, raw_norm = optim.clip_grad_norm(gradients, self.max_gradient_norm)
        checks = [mx.all(mx.isfinite(value)) for _, value in tree_flatten(gradients)]
        finite = mx.all(mx.stack(checks))
        mx.eval(loss, raw_norm, finite, gradients)
        loss_value = float(loss.item())
        norm_value = float(raw_norm.item())
        if (
            not bool(finite.item())
            or not math.isfinite(loss_value)
            or not math.isfinite(norm_value)
        ):
            raise RuntimeError("action-value loss or gradients became non-finite")
        self.optimizer.update(self.head, gradients)
        mx.eval(self.head.parameters(), self.optimizer.state)
        self.step += 1
        return loss_value, min(norm_value, self.max_gradient_norm), norm_value


def _network_config(payload: Mapping[str, object]) -> NetworkConfig:
    return NetworkConfig(
        trunk_channels=int(payload["trunk_channels"]),
        residual_blocks=int(payload["residual_blocks"]),
        policy_channels=int(payload["policy_channels"]),
        value_channels=int(payload["value_channels"]),
        value_hidden=int(payload["value_hidden"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head_snapshot(network: HarbiChessActionValueNetwork) -> tuple[tuple[str, mx.array], ...]:
    snapshot = tuple(
        (name, mx.array(value))
        for name, value in tree_flatten(network.action_value_head.parameters())
    )
    mx.eval([value for _, value in snapshot])
    return snapshot


def _prepare_data(
    records: tuple[ReplayRecord, ...],
    rows: Sequence[Mapping[str, object]],
    network: HarbiChessActionValueNetwork,
    *,
    rules: PythonChessRules,
) -> PreparedActionValueData:
    index = _record_index(records)
    encoder = BoardEncoder(rules)
    matched = []
    teacher_values = []
    legal_actions = []
    targets = []
    weights = []
    encoded = []
    for row in rows:
        key = (str(row["game_id"]), int(row["game_index"]), int(row["ply"]))
        record = index.get(key)
        if record is None:
            raise ValueError(f"TERAZI row is absent from replay: {key}")
        board = rules.board(record.state)
        q_low = dict(row["budgets"]["512"]["q"])
        q_high = dict(row["budgets"]["800"]["q"])
        visits_low = dict(row["budgets"]["512"]["visits"])
        visits_high = dict(row["budgets"]["800"]["visits"])
        common = q_low.keys() & q_high.keys()
        dense_target = [0.0] * POLICY_SIZE
        dense_weight = [0.0] * POLICY_SIZE
        sparse = {}
        raw_weights = {}
        for uci in common:
            action = move_to_action(board, chess.Move.from_uci(uci))
            low_count = int(visits_low[uci])
            high_count = int(visits_high[uci])
            label = (low_count * q_low[uci] + high_count * q_high[uci]) / (
                low_count + high_count
            )
            dense_target[action] = label
            sparse[action] = label
            raw_weights[action] = math.sqrt(min(low_count, high_count))
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            raise ValueError("action-value row has no common visited support")
        for action, value in raw_weights.items():
            dense_weight[action] = value / total_weight
        matched.append(record)
        teacher_values.append(sparse)
        legal_actions.append(
            tuple(sorted(move_to_action(board, move) for move in board.legal_moves))
        )
        targets.append(dense_target)
        weights.append(dense_weight)
        encoded.append(encoder.encode_state(record.state, board))
    shape = encoded[0].shape
    inputs = mx.array([position.values for position in encoded], dtype=mx.float32).reshape(
        len(encoded), *shape
    )
    trunk, state_values = network.frozen_action_features(inputs)
    target_array = mx.array(targets, dtype=mx.float32)
    weight_array = mx.array(weights, dtype=mx.float32)
    mx.eval(inputs, trunk, state_values, target_array, weight_array)
    return PreparedActionValueData(
        tuple(matched),
        inputs,
        trunk,
        state_values,
        target_array,
        weight_array,
        tuple(teacher_values),
        tuple(legal_actions),
    )


def _verified_values(
    data: PreparedActionValueData,
    *,
    rules: PythonChessRules,
    depth: int,
    workers: int,
) -> tuple[dict[int, float], ...]:
    verifier = ProcessTacticalOracle(TacticalOracleConfig(depth=depth), workers=workers)
    work = []
    try:
        for index, record in enumerate(data.records):
            board = rules.board(record.state)
            for action in data.legal_actions[index]:
                move = action_to_legal_move(board, action)
                child = rules.apply(record.state, ChessMove(move.uci()))
                work.append((index, action, child))

        def verify(item: tuple[int, int, object]):
            index, action, child = item
            return index, action, -verifier.value(child)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            items = tuple(pool.map(verify, work))
    finally:
        verifier.close()
    values: dict[int, dict[int, float]] = {}
    for index, action, value in items:
        values.setdefault(index, {})[action] = value
    return tuple(values[index] for index in range(len(data.records)))


def _policy_wdl_delta(
    base: HarbiChessNetwork,
    network: HarbiChessActionValueNetwork,
    inputs: mx.array,
) -> float:
    base_policy, base_wdl = base(inputs)
    policy, wdl = network(inputs)
    delta = mx.maximum(mx.max(mx.abs(policy - base_policy)), mx.max(mx.abs(wdl - base_wdl)))
    mx.eval(delta)
    return float(delta.item())


def _quality(
    network: HarbiChessActionValueNetwork,
    data: PreparedActionValueData,
    verified: tuple[dict[int, float], ...],
    *,
    base: HarbiChessNetwork,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    predictions = network.action_value_head(data.trunk, data.state_values)
    weighted_error = mx.sum(mx.square(predictions - data.targets) * data.weights) / mx.sum(
        data.weights
    )
    mx.eval(predictions, weighted_error)
    rows = predictions.tolist()
    correlations = []
    deltas = []
    harmful = 0
    regrets = []
    coverage = []
    for index, (record, legal, teacher, values) in enumerate(
        zip(data.records, data.legal_actions, data.teacher_values, verified, strict=True)
    ):
        predicted = {action: rows[index][action] for action in legal}
        correlations.append(_spearman(predicted, teacher))
        selected = min(legal, key=lambda action: (-predicted[action], action))
        raw_action = min(record.raw_policy, key=lambda item: (-item[1], item[0]))[0]
        delta = values[selected] - values[raw_action]
        deltas.append(delta)
        harmful += delta <= -0.025
        best = max(values.values())
        regrets.append(best - values[selected])
        top_sixteen = sorted(legal, key=lambda action: (-predicted[action], action))[:16]
        coverage.append(any(values[action] == best for action in top_sixteen))
    return {
        "weighted_q_mse": float(weighted_error.item()),
        "mean_teacher_q_spearman": mean(correlations),
        "mean_verified_delta_vs_raw": mean(deltas),
        "verified_delta_95_interval": _interval(
            tuple(deltas), samples=bootstrap_samples, seed=seed
        ),
        "harmful_count": harmful,
        "harmful_ratio": harmful / len(deltas),
        "mean_verified_regret": mean(regrets),
        "best_action_coverage_top_16": mean(coverage),
        "maximum_policy_wdl_logit_delta": _policy_wdl_delta(base, network, data.inputs),
    }


def _gate_reasons(
    quality: Mapping[str, object],
    *,
    baseline_mse: float,
    config: ActionValueTransferConfig,
    tactical: tuple[int, int],
    baseline_tactical: tuple[int, int],
    maximum_gradient_norm: float,
    maximum_unclipped_gradient_norm: float,
) -> tuple[str, ...]:
    reasons = []
    if float(quality["weighted_q_mse"]) > baseline_mse * (1 - config.minimum_mse_improvement):
        reasons.append("validation Q MSE did not improve by 20%")
    if float(quality["mean_teacher_q_spearman"]) < config.minimum_teacher_spearman:
        reasons.append("teacher-Q Spearman correlation is below 0.35")
    if float(quality["verified_delta_95_interval"][0]) <= 0:
        reasons.append("predicted-Q verified-improvement interval is not positive")
    if float(quality["harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("predicted-Q harmful-action ratio exceeds 10%")
    if float(quality["mean_verified_regret"]) > config.maximum_verified_regret:
        reasons.append("predicted-Q mean verified regret exceeds 0.10")
    if float(quality["best_action_coverage_top_16"]) < config.minimum_best_action_coverage:
        reasons.append("predicted-Q top-16 best-action coverage is below 80%")
    if float(quality["maximum_policy_wdl_logit_delta"]) > config.maximum_logit_delta:
        reasons.append("release policy/WDL logits changed")
    if tactical != baseline_tactical:
        reasons.append("release tactical solve counts changed")
    if (
        not math.isfinite(maximum_gradient_norm)
        or not math.isfinite(maximum_unclipped_gradient_norm)
        or maximum_gradient_norm > config.max_gradient_norm
    ):
        reasons.append("action-value gradient safety gate failed")
    return tuple(reasons)


def _clone_with_head(
    base_path: Path,
    network_config: NetworkConfig,
    head_weights: tuple[tuple[str, mx.array], ...],
    *,
    action_value_channels: int,
) -> HarbiChessActionValueNetwork:
    base = HarbiChessNetwork(network_config)
    base.load_weights(str(base_path))
    network = HarbiChessActionValueNetwork.from_base(
        base, action_value_channels=action_value_channels
    )
    network.action_value_head.load_weights(list(head_weights))
    mx.eval(network.parameters())
    return network


def run_action_value_transfer(config: ActionValueTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"action-value transfer output exists: {config.output_dir}")
    q_audit = json.loads(config.q_reliability_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if q_audit.get("gate", {}).get("passed"):
        raise ValueError("DEGER expects the failed TERAZI Q-target gate")
    rules = PythonChessRules()
    train_records = read_shard(config.train_shard, rules=rules).records
    validation_records = read_shard(config.validation_shard, rules=rules).records
    network_config = _network_config(run["config"])
    baseline_path = Path(run["baseline"]["path"])
    base = HarbiChessNetwork(network_config)
    base.load_weights(str(baseline_path))
    network = HarbiChessActionValueNetwork.from_base(
        base, action_value_channels=config.action_value_channels
    )
    train = _prepare_data(train_records, q_audit["rows"]["train"], network, rules=rules)
    validation = _prepare_data(
        validation_records, q_audit["rows"]["validation"], network, rules=rules
    )
    verified = _verified_values(
        validation,
        rules=rules,
        depth=config.verifier_depth,
        workers=config.verifier_workers,
    )
    learner = ActionValueLearner(
        network,
        learning_rate=config.learning_rate,
        max_gradient_norm=config.max_gradient_norm,
    )
    sampler = GameBalancedSampler(train.records, seed=config.seed)
    checkpoints: list[tuple[int, tuple[tuple[str, mx.array], ...], dict[str, object]]] = []
    maximum_gradient_norm = 0.0
    maximum_unclipped_gradient_norm = 0.0
    store = SnapshotStore(config.telemetry_path)
    dashboard = store.read()
    started = time.perf_counter()
    for step in range(config.steps + 1):
        if step in config.checkpoint_steps:
            quality = _quality(
                network,
                validation,
                verified,
                base=base,
                bootstrap_samples=config.bootstrap_samples,
                seed=config.bootstrap_seed + step,
            )
            checkpoints.append((step, _head_snapshot(network), quality))
            dashboard = replace(
                dashboard,
                updated_at=datetime.now(UTC).isoformat(),
                mode=RunMode.TRAINING if step < config.steps else RunMode.IDLE,
                mode_detail=f"DEGER action-value transfer · {step}/{config.steps} steps",
                pilot_status=PilotStatus.TRAINING,
                pilot_steps_planned=config.steps,
                pilot_steps_completed=step,
            )
            store.write_atomic(dashboard)
        if step == config.steps:
            break
        _, clipped_norm, raw_norm = learner.train_step(
            train.select(sampler.sample_indices(config.batch_size))
        )
        maximum_gradient_norm = max(maximum_gradient_norm, clipped_norm)
        maximum_unclipped_gradient_norm = max(maximum_unclipped_gradient_norm, raw_norm)

    baseline_tactical_payload = _tactical_metrics(
        base,
        network_config=network_config,
        budgets=(config.tactical_budget,),
        workers=config.tactical_workers,
        seed=config.seed,
    )
    baseline_tactical = (
        _tactical_solved(baseline_tactical_payload)[0],
        _tactical_solved(baseline_tactical_payload)[1][0],
    )
    baseline_mse = float(checkpoints[0][2]["weighted_q_mse"])
    rows = []
    eligible = []
    for step, head_weights, quality in checkpoints:
        candidate = _clone_with_head(
            baseline_path,
            network_config,
            head_weights,
            action_value_channels=config.action_value_channels,
        )
        tactical_payload = _tactical_metrics(
            candidate,
            network_config=network_config,
            budgets=(config.tactical_budget,),
            workers=config.tactical_workers,
            seed=config.seed,
        )
        tactical = (
            _tactical_solved(tactical_payload)[0],
            _tactical_solved(tactical_payload)[1][0],
        )
        reasons = (
            ("baseline control is not a trainable candidate",)
            if step == 0
            else _gate_reasons(
                quality,
                baseline_mse=baseline_mse,
                config=config,
                tactical=tactical,
                baseline_tactical=baseline_tactical,
                maximum_gradient_norm=maximum_gradient_norm,
                maximum_unclipped_gradient_norm=maximum_unclipped_gradient_norm,
            )
        )
        row = {
            "step": step,
            "quality": quality,
            "tactical": tactical_payload,
            "passed": not reasons,
            "reasons": reasons,
        }
        rows.append(row)
        if not reasons:
            eligible.append((float(quality["weighted_q_mse"]), step, head_weights, quality))

    checkpoint = None
    if eligible:
        _, selected_step, selected_weights, selected_quality = min(eligible)
        selected = _clone_with_head(
            baseline_path,
            network_config,
            selected_weights,
            action_value_channels=config.action_value_channels,
        )
        checkpoint_dir = config.output_dir / f"candidate-step-{selected_step:06d}"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "model.safetensors"
        temporary = checkpoint_dir / ".model.tmp.safetensors"
        selected.save_weights(str(temporary))
        os.replace(temporary, checkpoint_path)
        checkpoint = {
            "step": selected_step,
            "path": str(checkpoint_path),
            "model_sha256": _sha256(checkpoint_path),
            "quality": selected_quality,
            "completed_q_search_authorized": True,
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
                "q_reliability_result": str(config.q_reliability_result),
                "run_result": str(config.run_result),
                "train_shard": str(config.train_shard),
                "validation_shard": str(config.validation_shard),
                "output_dir": str(config.output_dir),
                "telemetry_path": str(config.telemetry_path),
            },
            "baseline": {
                "path": str(baseline_path),
                "model_sha256": run["baseline"]["model_sha256"],
                "weighted_q_mse": baseline_mse,
                "tactical": baseline_tactical_payload,
            },
            "training": {
                "maximum_gradient_norm": maximum_gradient_norm,
                "maximum_unclipped_gradient_norm": maximum_unclipped_gradient_norm,
                "elapsed_seconds": time.perf_counter() - started,
            },
            "checkpoints": rows,
            "passed": checkpoint is not None,
            "checkpoint": checkpoint,
            "completed_q_search_authorized": checkpoint is not None,
            "arena_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
        },
    )
    all_reasons = sorted({reason for row in rows for reason in row["reasons"]})
    dashboard = replace(
        dashboard,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "DEGER action-value transfer passed · completed-Q audit authorized"
            if checkpoint
            else "DEGER action-value transfer failed · learner remains blocked"
        ),
        pilot_status=PilotStatus.PASSED if checkpoint else PilotStatus.FAILED,
        pilot_steps_attempted=config.steps,
        pilot_steps_completed=config.steps,
        pilot_stop_reason="fixed_step_limit",
        pilot_stop_detail=(
            "Action-value transfer gate passed"
            if checkpoint
            else "; ".join(all_reasons)
        ),
        promotion_ready=False,
    )
    store.write_atomic(dashboard)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-reliability-result", required=True, type=Path)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    path = run_action_value_transfer(
        ActionValueTransferConfig(
            q_reliability_result=arguments.q_reliability_result,
            run_result=arguments.run_result,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            output_dir=arguments.output_dir,
            telemetry_path=arguments.telemetry,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
