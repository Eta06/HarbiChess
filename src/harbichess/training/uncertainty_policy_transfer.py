"""Transfer uncertainty-qualified search targets into a low-rank policy update."""

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
from statistics import mean

import chess
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import POLICY_SIZE, move_to_action
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.consensus_target import _record_index
from harbichess.evaluation.search_q_reliability import _spearman
from harbichess.evaluation.teacher_qualification import _atomic_json, _interval, _source_commit
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.training.batch import GameBalancedSampler
from harbichess.training.learner_transfer import _tactical_metrics, _tactical_solved


@dataclass(frozen=True, slots=True)
class UncertaintyPolicyTransferConfig:
    label_result: Path
    dataset_result: Path
    run_result: Path
    train_shard: Path
    validation_shard: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    rank: int = 8
    steps: int = 480
    batch_size: int = 16
    learning_rate: float = 2e-4
    checkpoint_steps: tuple[int, ...] = (0, 60, 120, 240, 480)
    max_gradient_norm: float = 5.0
    bootstrap_samples: int = 2_000
    seed: int = 2026082833
    minimum_cross_entropy_improvement: float = 0.05
    minimum_teacher_spearman: float = 0.35
    maximum_harmful_ratio: float = 0.10
    maximum_verified_regret: float = 0.10
    minimum_best_action_coverage: float = 0.80
    tactical_budgets: tuple[int, ...] = (64, 512)
    tactical_workers: int = 8

    def __post_init__(self) -> None:
        if (
            min(
                self.rank,
                self.steps,
                self.batch_size,
                self.bootstrap_samples,
                self.seed,
                self.tactical_workers,
            )
            <= 0
            or self.learning_rate <= 0
            or self.max_gradient_norm <= 0
            or self.checkpoint_steps != tuple(sorted(set(self.checkpoint_steps)))
            or self.checkpoint_steps[0] != 0
            or self.checkpoint_steps[-1] != self.steps
            or not 0 <= self.minimum_cross_entropy_improvement < 1
            or not -1 <= self.minimum_teacher_spearman <= 1
            or not 0 <= self.maximum_harmful_ratio <= 1
            or self.maximum_verified_regret < 0
            or not 0 <= self.minimum_best_action_coverage <= 1
            or not self.tactical_budgets
            or any(budget <= 0 for budget in self.tactical_budgets)
        ):
            raise ValueError("uncertainty policy transfer configuration is invalid")


@dataclass(frozen=True, slots=True)
class PreparedPolicyData:
    records: tuple[ReplayRecord, ...]
    inputs: mx.array
    features: mx.array
    base_logits: mx.array
    targets: mx.array
    legal_masks: mx.array
    teacher: tuple[dict[int, float], ...]
    verified: tuple[dict[int, float], ...]
    legal_actions: tuple[tuple[int, ...], ...]

    def select(self, indices: tuple[int, ...]) -> tuple[mx.array, ...]:
        rows = mx.array(indices, dtype=mx.int32)
        return tuple(
            mx.take(array, rows, axis=0)
            for array in (self.features, self.base_logits, self.targets, self.legal_masks)
        )


class LowRankPolicyAdapter(nn.Module):
    """Function-preserving low-rank update that can be merged into policy_linear."""

    def __init__(self, feature_size: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(feature_size, rank, bias=False)
        self.up = nn.Linear(rank, POLICY_SIZE, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)

    def __call__(self, features: mx.array, base_logits: mx.array) -> mx.array:
        return base_logits + self.up(self.down(features))

    def merged_weight(self, base_weight: mx.array) -> mx.array:
        return base_weight + self.up.weight @ self.down.weight


class PolicyAdapterLearner:
    def __init__(
        self, adapter: LowRankPolicyAdapter, *, learning_rate: float, max_gradient_norm: float
    ) -> None:
        self.adapter = adapter
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
        self.max_gradient_norm = max_gradient_norm
        self._loss_and_grad = nn.value_and_grad(adapter, self._loss)

    def _loss(
        self,
        features: mx.array,
        base_logits: mx.array,
        targets: mx.array,
        legal_masks: mx.array,
    ) -> mx.array:
        logits = mx.where(
            legal_masks,
            self.adapter(features, base_logits),
            mx.array(-1e9),
        )
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    def train_step(self, batch: tuple[mx.array, ...]) -> tuple[float, float]:
        loss, gradients = self._loss_and_grad(*batch)
        gradients, raw_norm = optim.clip_grad_norm(gradients, self.max_gradient_norm)
        finite = mx.all(
            mx.stack([mx.all(mx.isfinite(value)) for _, value in tree_flatten(gradients)])
        )
        mx.eval(loss, raw_norm, finite, gradients)
        loss_value = float(loss.item())
        norm_value = float(raw_norm.item())
        if not bool(finite.item()) or not all(map(math.isfinite, (loss_value, norm_value))):
            raise RuntimeError("policy adapter loss or gradients became non-finite")
        self.optimizer.update(self.adapter, gradients)
        mx.eval(self.adapter.parameters(), self.optimizer.state)
        return loss_value, norm_value


def _network_config(payload: Mapping[str, object]) -> NetworkConfig:
    return NetworkConfig(
        trunk_channels=int(payload["trunk_channels"]),
        residual_blocks=int(payload["residual_blocks"]),
        policy_channels=int(payload["policy_channels"]),
        value_channels=int(payload["value_channels"]),
        value_hidden=int(payload["value_hidden"]),
    )


def _dense_target(
    board: chess.Board, labels: Sequence[Sequence[object]]
) -> tuple[list[float], list[bool], dict[int, float], tuple[int, ...]]:
    targets = [0.0] * POLICY_SIZE
    legal = tuple(sorted(move_to_action(board, move) for move in board.legal_moves))
    legal_mask = [False] * POLICY_SIZE
    for action in legal:
        legal_mask[action] = True
    teacher = {}
    for uci, _q, _drift, weight in labels:
        action = move_to_action(board, chess.Move.from_uci(str(uci)))
        targets[action] = float(weight)
        if float(weight) > 0:
            teacher[action] = float(weight)
    if not math.isclose(sum(targets), 1.0, abs_tol=1e-6):
        raise ValueError("uncertainty policy target must sum to one")
    return targets, legal_mask, teacher, legal


def _prepare_data(
    records: tuple[ReplayRecord, ...],
    label_rows: Sequence[Mapping[str, object]],
    verifier_rows: Sequence[Mapping[str, object]],
    network: HarbiChessNetwork,
) -> PreparedPolicyData:
    rules = PythonChessRules()
    index = _record_index(records)
    verifier = {
        (str(row["game_id"]), int(row["game_index"]), int(row["ply"])): row
        for row in verifier_rows
    }
    encoder = BoardEncoder(rules)
    matched = []
    encoded = []
    targets = []
    masks = []
    teachers = []
    verified = []
    legal_rows = []
    for row in label_rows:
        key = (str(row["game_id"]), int(row["game_index"]), int(row["ply"]))
        record = index.get(key)
        verifier_row = verifier.get(key)
        if record is None or verifier_row is None:
            raise ValueError(f"AKIS row is absent from replay or verifier data: {key}")
        board = rules.board(record.state)
        target, mask, teacher, legal = _dense_target(board, row["labels"])
        matched.append(record)
        encoded.append(encoder.encode_state(record.state, board))
        targets.append(target)
        masks.append(mask)
        teachers.append(teacher)
        legal_rows.append(legal)
        verified.append(
            {
                move_to_action(board, chess.Move.from_uci(str(uci))): float(value)
                for uci, value in verifier_row["verified_values"]
            }
        )
    shape = encoded[0].shape
    inputs = mx.array([position.values for position in encoded], dtype=mx.float32).reshape(
        len(encoded), *shape
    )
    trunk = network._trunk(inputs)
    features = mx.stop_gradient(network._policy_features(trunk))
    base_logits = mx.stop_gradient(network.policy_linear(features))
    target_array = mx.array(targets, dtype=mx.float32)
    mask_array = mx.array(masks, dtype=mx.bool_)
    mx.eval(inputs, features, base_logits, target_array, mask_array)
    return PreparedPolicyData(
        tuple(matched),
        inputs,
        features,
        base_logits,
        target_array,
        mask_array,
        tuple(teachers),
        tuple(verified),
        tuple(legal_rows),
    )


def _snapshot(adapter: LowRankPolicyAdapter) -> tuple[tuple[str, mx.array], ...]:
    result = tuple((name, mx.array(value)) for name, value in tree_flatten(adapter.parameters()))
    mx.eval([value for _, value in result])
    return result


def _clone_adapter(
    feature_size: int, rank: int, weights: tuple[tuple[str, mx.array], ...]
) -> LowRankPolicyAdapter:
    adapter = LowRankPolicyAdapter(feature_size, rank)
    adapter.load_weights(list(weights))
    mx.eval(adapter.parameters())
    return adapter


def _quality(
    adapter: LowRankPolicyAdapter,
    data: PreparedPolicyData,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    logits = mx.where(
        data.legal_masks,
        adapter(data.features, data.base_logits),
        mx.array(-1e9),
    )
    log_probs = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    cross_entropy = -mx.sum(data.targets * log_probs) / data.targets.shape[0]
    mx.eval(logits, log_probs, cross_entropy)
    rows = logits.tolist()
    correlations = []
    deltas = []
    harmful = 0
    regrets = []
    coverage = []
    for index, (record, legal, teacher, values) in enumerate(
        zip(data.records, data.legal_actions, data.teacher, data.verified, strict=True)
    ):
        prediction = {action: rows[index][action] for action in teacher}
        correlations.append(_spearman(prediction, teacher))
        selected = min(legal, key=lambda action: (-rows[index][action], action))
        raw_action = min(record.raw_policy, key=lambda item: (-item[1], item[0]))[0]
        delta = values[selected] - values[raw_action]
        deltas.append(delta)
        harmful += delta <= -0.025
        best = max(values.values())
        regrets.append(best - values[selected])
        top_sixteen = sorted(legal, key=lambda action: (-rows[index][action], action))[:16]
        coverage.append(any(values[action] == best for action in top_sixteen))
    return {
        "uncertainty_policy_cross_entropy": float(cross_entropy.item()),
        "mean_teacher_policy_spearman": mean(correlations),
        "mean_verified_delta_vs_raw": mean(deltas),
        "verified_delta_95_interval": _interval(
            tuple(deltas), samples=bootstrap_samples, seed=seed
        ),
        "harmful_count": harmful,
        "harmful_ratio": harmful / len(deltas),
        "mean_verified_regret": mean(regrets),
        "best_action_coverage_top_16": mean(coverage),
    }


def _candidate_reasons(
    quality: Mapping[str, object],
    *,
    baseline_cross_entropy: float,
    config: UncertaintyPolicyTransferConfig,
    tactical: tuple[int, tuple[int, ...]],
    baseline_tactical: tuple[int, tuple[int, ...]],
    wdl_delta: float,
    maximum_gradient_norm: float,
) -> tuple[str, ...]:
    reasons = []
    if float(quality["uncertainty_policy_cross_entropy"]) > baseline_cross_entropy * (
        1 - config.minimum_cross_entropy_improvement
    ):
        reasons.append("uncertainty-policy cross entropy did not improve by 5%")
    if float(quality["mean_teacher_policy_spearman"]) < config.minimum_teacher_spearman:
        reasons.append("teacher-policy Spearman correlation is below 0.35")
    if float(quality["verified_delta_95_interval"][0]) <= 0:
        reasons.append("policy verified-improvement interval is not positive")
    if float(quality["harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("policy harmful-action ratio exceeds 10%")
    if float(quality["mean_verified_regret"]) > config.maximum_verified_regret:
        reasons.append("policy mean verified regret exceeds 0.10")
    if float(quality["best_action_coverage_top_16"]) < config.minimum_best_action_coverage:
        reasons.append("policy top-16 best-action coverage is below 80%")
    if wdl_delta != 0.0:
        reasons.append("WDL logits changed")
    if tactical[0] < baseline_tactical[0] or any(
        value < baseline
        for value, baseline in zip(tactical[1], baseline_tactical[1], strict=True)
    ):
        reasons.append("raw-policy or search tactical solve count regressed")
    if not math.isfinite(maximum_gradient_norm) or maximum_gradient_norm > config.max_gradient_norm:
        reasons.append("gradient safety limit was exceeded")
    return tuple(reasons)


def _merged_network(
    baseline_path: Path,
    network_config: NetworkConfig,
    adapter: LowRankPolicyAdapter,
) -> HarbiChessNetwork:
    network = HarbiChessNetwork(network_config)
    network.load_weights(str(baseline_path))
    network.policy_linear.weight = adapter.merged_weight(network.policy_linear.weight)
    mx.eval(network.parameters())
    return network


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_uncertainty_policy_transfer(config: UncertaintyPolicyTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"uncertainty policy output exists: {config.output_dir}")
    labels = json.loads(config.label_result.read_text(encoding="utf-8"))
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if not labels.get("gate", {}).get("spatial_transfer_authorized"):
        raise ValueError("AKIS requires qualified uncertainty labels")
    network_config = _network_config(run["config"])
    baseline_path = Path(run["baseline"]["path"])
    base = HarbiChessNetwork(network_config)
    base.load_weights(str(baseline_path))
    rules = PythonChessRules()
    train_records = read_shard(config.train_shard, rules=rules).records
    validation_records = read_shard(config.validation_shard, rules=rules).records
    train = _prepare_data(
        train_records, labels["rows"]["train"], dataset["rows"]["train"], base
    )
    validation = _prepare_data(
        validation_records,
        labels["rows"]["validation"],
        dataset["rows"]["validation"],
        base,
    )
    feature_size = int(train.features.shape[1])
    mx.random.seed(config.seed)
    adapter = LowRankPolicyAdapter(feature_size, config.rank)
    learner = PolicyAdapterLearner(
        adapter,
        learning_rate=config.learning_rate,
        max_gradient_norm=config.max_gradient_norm,
    )
    sampler = GameBalancedSampler(train.records, seed=config.seed)
    checkpoints = []
    maximum_gradient_norm = 0.0
    store = SnapshotStore(config.telemetry_path)
    dashboard = store.read()
    started = time.perf_counter()
    for step in range(config.steps + 1):
        if step in config.checkpoint_steps:
            checkpoints.append(
                (
                    step,
                    _snapshot(adapter),
                    _quality(
                        adapter,
                        train,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed + step + 10_000,
                    ),
                    _quality(
                        adapter,
                        validation,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed + step,
                    ),
                )
            )
            dashboard = replace(
                dashboard,
                updated_at=datetime.now(UTC).isoformat(),
                mode=RunMode.TRAINING if step < config.steps else RunMode.IDLE,
                mode_detail=f"AKIS uncertainty-policy transfer · {step}/{config.steps}",
                pilot_status=PilotStatus.TRAINING,
                pilot_steps_planned=config.steps,
                pilot_steps_completed=step,
            )
            store.write_atomic(dashboard)
        if step == config.steps:
            break
        _, gradient_norm = learner.train_step(
            train.select(sampler.sample_indices(config.batch_size))
        )
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)

    baseline_tactical_payload = _tactical_metrics(
        base,
        network_config=network_config,
        budgets=config.tactical_budgets,
        workers=config.tactical_workers,
        seed=config.seed,
    )
    baseline_tactical = _tactical_solved(baseline_tactical_payload)
    baseline_ce = float(checkpoints[0][3]["uncertainty_policy_cross_entropy"])
    evaluated = []
    eligible = []
    base_wdl = base(validation.inputs)[1]
    mx.eval(base_wdl)
    for step, weights, train_quality, quality in checkpoints:
        candidate_adapter = _clone_adapter(feature_size, config.rank, weights)
        candidate = _merged_network(baseline_path, network_config, candidate_adapter)
        tactical_payload = _tactical_metrics(
            candidate,
            network_config=network_config,
            budgets=config.tactical_budgets,
            workers=config.tactical_workers,
            seed=config.seed,
        )
        tactical = _tactical_solved(tactical_payload)
        candidate_wdl = candidate(validation.inputs)[1]
        wdl_delta_array = mx.max(mx.abs(candidate_wdl - base_wdl))
        mx.eval(wdl_delta_array)
        wdl_delta = float(wdl_delta_array.item())
        reasons = (
            ("baseline control is not a trainable candidate",)
            if step == 0
            else _candidate_reasons(
                quality,
                baseline_cross_entropy=baseline_ce,
                config=config,
                tactical=tactical,
                baseline_tactical=baseline_tactical,
                wdl_delta=wdl_delta,
                maximum_gradient_norm=maximum_gradient_norm,
            )
        )
        evaluated.append(
            {
                "step": step,
                "train_quality": train_quality,
                "quality": quality,
                "maximum_wdl_logit_delta": wdl_delta,
                "tactical": tactical_payload,
                "passed": not reasons,
                "reasons": reasons,
            }
        )
        if not reasons:
            eligible.append((float(quality["uncertainty_policy_cross_entropy"]), step, candidate))

    checkpoint = None
    if eligible:
        _, selected_step, selected = min(eligible)
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
            "search_qualification_authorized": True,
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
                **{
                    name: str(getattr(config, name))
                    for name in (
                        "label_result",
                        "dataset_result",
                        "run_result",
                        "train_shard",
                        "validation_shard",
                        "output_dir",
                        "telemetry_path",
                    )
                },
            },
            "baseline": {
                "path": str(baseline_path),
                "model_sha256": run["baseline"]["model_sha256"],
                "uncertainty_policy_cross_entropy": baseline_ce,
                "tactical": baseline_tactical_payload,
            },
            "training": {
                "elapsed_seconds": time.perf_counter() - started,
                "maximum_gradient_norm": maximum_gradient_norm,
            },
            "checkpoints": evaluated,
            "passed": checkpoint is not None,
            "checkpoint": checkpoint,
            "search_qualification_authorized": checkpoint is not None,
            "arena_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
        },
    )
    reasons = sorted({reason for row in evaluated for reason in row["reasons"]})
    dashboard = replace(
        dashboard,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            "AKIS policy transfer passed · search qualification authorized"
            if checkpoint
            else "AKIS policy transfer failed · learner remains blocked"
        ),
        pilot_status=PilotStatus.PASSED if checkpoint else PilotStatus.FAILED,
        pilot_steps_attempted=config.steps,
        pilot_steps_completed=config.steps,
        pilot_stop_reason="fixed_step_limit",
        pilot_stop_detail=("AKIS gate passed" if checkpoint else "; ".join(reasons)),
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
    parser.add_argument("--seed", type=int, default=2026082833)
    arguments = parser.parse_args(argv)
    path = run_uncertainty_policy_transfer(
        UncertaintyPolicyTransferConfig(
            label_result=arguments.label_result,
            dataset_result=arguments.dataset_result,
            run_result=arguments.run_result,
            train_shard=arguments.train_shard,
            validation_shard=arguments.validation_shard,
            output_dir=arguments.output_dir,
            telemetry_path=arguments.telemetry,
            seed=arguments.seed,
        )
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
