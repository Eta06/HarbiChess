"""Run the frozen OLCEK uncertainty-weighted spatial-Q transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import chess
import mlx.core as mx

from harbichess.backends.action_value_network import HarbiChessSpatialActionValueNetwork
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import POLICY_SIZE, move_to_action
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.consensus_target import _record_index
from harbichess.evaluation.teacher_qualification import _atomic_json, _source_commit
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.training.action_value_transfer import (
    ActionValueLearner,
    PreparedActionValueData,
    _gate_reasons,
    _head_snapshot,
    _quality,
)
from harbichess.training.batch import GameBalancedSampler
from harbichess.training.learner_transfer import _tactical_metrics, _tactical_solved


@dataclass(frozen=True, slots=True)
class SpatialActionValueTransferConfig:
    label_result: Path
    dataset_result: Path
    run_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    steps: int = 480
    batch_size: int = 16
    learning_rate: float = 2e-4
    checkpoint_steps: tuple[int, ...] = (0, 60, 120, 240, 480)
    max_gradient_norm: float = 5.0
    bootstrap_samples: int = 2_000
    seed: int = 2026082829
    bootstrap_seed: int = 2026082826
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
                self.steps,
                self.batch_size,
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
            raise ValueError("spatial action-value transfer configuration is invalid")


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


def _dense_labels(
    board: chess.Board, labels: Sequence[Sequence[object]]
) -> tuple[list[float], list[float], dict[int, float]]:
    targets = [0.0] * POLICY_SIZE
    weights = [0.0] * POLICY_SIZE
    sparse = {}
    for uci, target, _drift, weight in labels:
        action = move_to_action(board, chess.Move.from_uci(str(uci)))
        targets[action] = float(target)
        weights[action] = float(weight)
        if float(weight) > 0:
            sparse[action] = float(target)
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError("uncertainty label weights must sum to one")
    return targets, weights, sparse


def _prepare_data(
    records: tuple[ReplayRecord, ...],
    rows: Sequence[Mapping[str, object]],
    network: HarbiChessSpatialActionValueNetwork,
    *,
    rules: PythonChessRules,
) -> PreparedActionValueData:
    index = _record_index(records)
    encoder = BoardEncoder(rules)
    matched = []
    encoded = []
    targets = []
    weights = []
    teacher_values = []
    legal_actions = []
    for row in rows:
        key = (str(row["game_id"]), int(row["game_index"]), int(row["ply"]))
        record = index.get(key)
        if record is None:
            raise ValueError(f"OLCEK row is absent from replay: {key}")
        board = rules.board(record.state)
        dense_targets, dense_weights, sparse = _dense_labels(board, row["labels"])
        matched.append(record)
        encoded.append(encoder.encode_state(record.state, board))
        targets.append(dense_targets)
        weights.append(dense_weights)
        teacher_values.append(sparse)
        legal_actions.append(
            tuple(sorted(move_to_action(board, move) for move in board.legal_moves))
        )
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
    rows: Sequence[Mapping[str, object]], data: PreparedActionValueData
) -> tuple[dict[int, float], ...]:
    by_key = {
        (str(row["game_id"]), int(row["game_index"]), int(row["ply"])): row
        for row in rows
    }
    rules = PythonChessRules()
    output = []
    for record in data.records:
        key = (record.game_id, record.game_index, record.ply)
        row = by_key.get(key)
        if row is None:
            raise ValueError(f"OLCEK verifier row is absent: {key}")
        board = rules.board(record.state)
        output.append(
            {
                move_to_action(board, chess.Move.from_uci(str(uci))): float(value)
                for uci, value in row["verified_values"]
            }
        )
    return tuple(output)


def _clone_with_head(
    baseline_path: Path,
    network_config: NetworkConfig,
    head_weights: tuple[tuple[str, mx.array], ...],
) -> HarbiChessSpatialActionValueNetwork:
    base = HarbiChessNetwork(network_config)
    base.load_weights(str(baseline_path))
    network = HarbiChessSpatialActionValueNetwork.from_base(base)
    network.action_value_head.load_weights(list(head_weights))
    mx.eval(network.parameters())
    return network


def run_spatial_action_value_transfer(config: SpatialActionValueTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"spatial action-value output exists: {config.output_dir}")
    labels = json.loads(config.label_result.read_text(encoding="utf-8"))
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if not labels.get("gate", {}).get("spatial_transfer_authorized"):
        raise ValueError("spatial transfer requires qualified uncertainty labels")
    if Path(labels["config"]["dataset_result"]) != config.dataset_result:
        raise ValueError("uncertainty labels do not match the supplied Q dataset")
    rules = PythonChessRules()
    train_records = read_shard(config.train_shard, rules=rules).records
    validation_records = read_shard(config.validation_shard, rules=rules).records
    network_config = _network_config(run["config"])
    baseline_path = Path(run["baseline"]["path"])
    base = HarbiChessNetwork(network_config)
    base.load_weights(str(baseline_path))
    network = HarbiChessSpatialActionValueNetwork.from_base(base)
    train = _prepare_data(train_records, labels["rows"]["train"], network, rules=rules)
    validation = _prepare_data(
        validation_records, labels["rows"]["validation"], network, rules=rules
    )
    verified = _verified_values(dataset["rows"]["validation"], validation)
    learner = ActionValueLearner(
        network,
        learning_rate=config.learning_rate,
        max_gradient_norm=config.max_gradient_norm,
    )
    sampler = GameBalancedSampler(train.records, seed=config.seed)
    checkpoints = []
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
                mode_detail=f"OLCEK spatial-Q transfer · {step}/{config.steps} steps",
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
    baseline_raw, baseline_search = _tactical_solved(baseline_tactical_payload)
    baseline_tactical = (baseline_raw, baseline_search[0])
    baseline_mse = float(checkpoints[0][2]["weighted_q_mse"])
    rows = []
    eligible = []
    for step, head_weights, quality in checkpoints:
        candidate = _clone_with_head(baseline_path, network_config, head_weights)
        tactical_payload = _tactical_metrics(
            candidate,
            network_config=network_config,
            budgets=(config.tactical_budget,),
            workers=config.tactical_workers,
            seed=config.seed,
        )
        candidate_raw, candidate_search = _tactical_solved(tactical_payload)
        tactical = (candidate_raw, candidate_search[0])
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
        rows.append(
            {
                "step": step,
                "quality": quality,
                "tactical": tactical_payload,
                "passed": not reasons,
                "reasons": reasons,
            }
        )
        if not reasons:
            eligible.append((float(quality["weighted_q_mse"]), step, head_weights, quality))

    checkpoint = None
    if eligible:
        _, selected_step, selected_weights, selected_quality = min(eligible)
        selected = _clone_with_head(baseline_path, network_config, selected_weights)
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
                "label_result": str(config.label_result),
                "dataset_result": str(config.dataset_result),
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
            "OLCEK spatial-Q transfer passed · completed-Q audit authorized"
            if checkpoint
            else "OLCEK spatial-Q transfer failed · learner remains blocked"
        ),
        pilot_status=PilotStatus.PASSED if checkpoint else PilotStatus.FAILED,
        pilot_steps_attempted=config.steps,
        pilot_steps_completed=config.steps,
        pilot_stop_reason="fixed_step_limit",
        pilot_stop_detail=(
            "Spatial-Q transfer gate passed" if checkpoint else "; ".join(all_reasons)
        ),
        promotion_ready=False,
    )
    store.write_atomic(dashboard)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-result", required=True, type=Path)
    parser.add_argument("--dataset-result", required=True, type=Path)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--validation-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    path = run_spatial_action_value_transfer(
        SpatialActionValueTransferConfig(
            label_result=arguments.label_result,
            dataset_result=arguments.dataset_result,
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
