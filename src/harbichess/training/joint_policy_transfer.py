"""Test joint trunk-policy learning under broad policy and WDL distillation."""

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

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.chess.rules import PythonChessRules
from harbichess.dashboard.state import PilotStatus, RunMode, SnapshotStore
from harbichess.evaluation.teacher_qualification import _atomic_json, _source_commit
from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import read_shard
from harbichess.training.anchored_policy_transfer import (
    _hash_key,
    _reasons,
    _split_games,
)
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.learner import MLXLearner
from harbichess.training.policy_projection import _target_entropy
from harbichess.training.uncertainty_policy_transfer import (
    _network_config,
    _policy_quality,
    _prepare_data,
)


@dataclass(frozen=True, slots=True)
class JointPolicyTransferConfig:
    policy_target_result: Path
    dataset_result: Path
    run_result: Path
    train_shard: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    holdout_fraction: float = 0.20
    anchor_positions: int = 2_048
    steps: int = 960
    target_batch_size: int = 16
    anchor_batch_size: int = 64
    learning_rate: float = 2e-4
    policy_anchor_weights: tuple[float, ...] = (1.0, 4.0, 16.0)
    wdl_anchor_weight: float = 4.0
    split_seed: int = 2026082852
    arm_seed: int = 2026082856
    max_gradient_norm: float = 5.0
    bootstrap_samples: int = 2_000
    minimum_gap_fraction: float = 0.20
    minimum_teacher_spearman: float = 0.35
    maximum_harmful_ratio: float = 0.10
    maximum_verified_regret: float = 0.10
    minimum_best_action_coverage: float = 0.80
    maximum_policy_anchor_kl: float = 0.02
    maximum_wdl_anchor_kl: float = 0.002
    maximum_expected_score_drift: float = 0.02

    def __post_init__(self) -> None:
        if (
            min(
                self.anchor_positions,
                self.steps,
                self.target_batch_size,
                self.anchor_batch_size,
                self.split_seed,
                self.arm_seed,
                self.bootstrap_samples,
            )
            <= 0
            or not 0 < self.holdout_fraction < 1
            or min(self.learning_rate, self.wdl_anchor_weight, self.max_gradient_norm) <= 0
            or not self.policy_anchor_weights
            or any(weight <= 0 for weight in self.policy_anchor_weights)
            or min(
                self.maximum_policy_anchor_kl,
                self.maximum_wdl_anchor_kl,
                self.maximum_expected_score_drift,
            )
            < 0
        ):
            raise ValueError("joint policy transfer configuration is invalid")

    @property
    def maximum_anchor_kl(self) -> float:
        """Expose the shared CIPA policy-anchor gate without changing its limit."""

        return self.maximum_policy_anchor_kl


@dataclass(frozen=True, slots=True)
class JointAnchorData:
    records: tuple[ReplayRecord, ...]
    inputs: mx.array
    base_policy_logits: mx.array
    base_wdl_logits: mx.array
    legal_masks: mx.array

    def select(self, indices: tuple[int, ...]) -> tuple[mx.array, ...]:
        rows = mx.array(indices, dtype=mx.int32)
        return tuple(
            mx.take(array, rows, axis=0)
            for array in (
                self.inputs,
                self.base_policy_logits,
                self.base_wdl_logits,
                self.legal_masks,
            )
        )


class JointPolicyLearner:
    def __init__(
        self,
        network: HarbiChessNetwork,
        *,
        learning_rate: float,
        policy_anchor_weight: float,
        wdl_anchor_weight: float,
        max_gradient_norm: float,
    ) -> None:
        self.network = network
        self.policy_anchor_weight = policy_anchor_weight
        self.wdl_anchor_weight = wdl_anchor_weight
        self.max_gradient_norm = max_gradient_norm
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
        self._loss_and_grad = nn.value_and_grad(network, self._loss)

    def _loss(
        self,
        target_inputs: mx.array,
        targets: mx.array,
        target_masks: mx.array,
        anchor_inputs: mx.array,
        base_policy_logits: mx.array,
        base_wdl_logits: mx.array,
        anchor_masks: mx.array,
    ) -> mx.array:
        target_logits, _ = self.network(target_inputs)
        target_logits = mx.where(target_masks, target_logits, mx.array(-1e9))
        target_loss = nn.losses.cross_entropy(target_logits, targets, reduction="mean")

        candidate_policy, candidate_wdl = self.network(anchor_inputs)
        base_policy = mx.where(anchor_masks, base_policy_logits, mx.array(-1e9))
        candidate_policy = mx.where(anchor_masks, candidate_policy, mx.array(-1e9))
        base_policy_log = base_policy - mx.logsumexp(base_policy, axis=1, keepdims=True)
        candidate_policy_log = candidate_policy - mx.logsumexp(
            candidate_policy, axis=1, keepdims=True
        )
        base_policy_probability = mx.exp(base_policy_log)
        policy_kl = mx.mean(
            mx.sum(
                base_policy_probability * (base_policy_log - candidate_policy_log),
                axis=1,
            )
        )

        base_wdl_log = base_wdl_logits - mx.logsumexp(base_wdl_logits, axis=1, keepdims=True)
        candidate_wdl_log = candidate_wdl - mx.logsumexp(candidate_wdl, axis=1, keepdims=True)
        wdl_kl = mx.mean(mx.sum(mx.exp(base_wdl_log) * (base_wdl_log - candidate_wdl_log), axis=1))
        return target_loss + self.policy_anchor_weight * policy_kl + self.wdl_anchor_weight * wdl_kl

    def train_step(self, arrays: tuple[mx.array, ...]) -> tuple[float, float]:
        loss, gradients = self._loss_and_grad(*arrays)
        gradients, raw_norm = optim.clip_grad_norm(gradients, self.max_gradient_norm)
        finite = mx.all(
            mx.stack([mx.all(mx.isfinite(value)) for _, value in tree_flatten(gradients)])
        )
        mx.eval(loss, raw_norm, finite, gradients)
        loss_value = float(loss.item())
        norm_value = float(raw_norm.item())
        if not bool(finite.item()) or not all(map(math.isfinite, (loss_value, norm_value))):
            raise RuntimeError("joint policy loss or gradients became non-finite")
        self.optimizer.update(self.network, gradients)
        mx.eval(self.network.parameters(), self.optimizer.state)
        return loss_value, norm_value


def _anchor_data(records: tuple[ReplayRecord, ...], baseline: HarbiChessNetwork) -> JointAnchorData:
    batch = MLXLearner.prepare_batch(build_training_batch(records))
    policy, wdl = baseline(batch.inputs)
    policy = mx.stop_gradient(policy)
    wdl = mx.stop_gradient(wdl)
    mx.eval(batch.inputs, policy, wdl, batch.legal_masks)
    return JointAnchorData(records, batch.inputs, policy, wdl, batch.legal_masks)


def _masked_kl(
    base_logits: mx.array,
    candidate_logits: mx.array,
    masks: mx.array | None = None,
) -> float:
    if masks is not None:
        base_logits = mx.where(masks, base_logits, mx.array(-1e9))
        candidate_logits = mx.where(masks, candidate_logits, mx.array(-1e9))
    base_log = base_logits - mx.logsumexp(base_logits, axis=1, keepdims=True)
    candidate_log = candidate_logits - mx.logsumexp(candidate_logits, axis=1, keepdims=True)
    value = mx.mean(mx.sum(mx.exp(base_log) * (base_log - candidate_log), axis=1))
    mx.eval(value)
    return float(value.item())


def _anchor_drift(
    network: HarbiChessNetwork, anchor: JointAnchorData
) -> tuple[float, float, float]:
    policy, wdl = network(anchor.inputs)
    policy_kl = _masked_kl(anchor.base_policy_logits, policy, anchor.legal_masks)
    wdl_kl = _masked_kl(anchor.base_wdl_logits, wdl)
    base_probability = mx.softmax(anchor.base_wdl_logits, axis=1)
    candidate_probability = mx.softmax(wdl, axis=1)
    base_score = base_probability[:, 0] - base_probability[:, 2]
    candidate_score = candidate_probability[:, 0] - candidate_probability[:, 2]
    score_drift = mx.mean(mx.abs(candidate_score - base_score))
    mx.eval(score_drift)
    return policy_kl, wdl_kl, float(score_drift.item())


def _joint_reasons(
    quality: Mapping[str, object],
    *,
    gap_fraction: float,
    policy_kl: float,
    wdl_kl: float,
    expected_score_drift: float,
    maximum_gradient_norm: float,
    config: JointPolicyTransferConfig,
) -> tuple[str, ...]:
    shared = _reasons(
        quality,
        gap_fraction=gap_fraction,
        anchor_kl=policy_kl,
        maximum_gradient_norm=maximum_gradient_norm,
        config=config,  # type: ignore[arg-type]
    )
    reasons = list(shared)
    if wdl_kl > config.maximum_wdl_anchor_kl:
        reasons.append("broad replay WDL KL exceeds 0.002")
    if expected_score_drift > config.maximum_expected_score_drift:
        reasons.append("broad replay expected-score drift exceeds 0.02")
    return tuple(reasons)


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_joint_policy_transfer(config: JointPolicyTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"joint transfer output exists: {config.output_dir}")
    target = json.loads(config.policy_target_result.read_text(encoding="utf-8"))
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if not target.get("gate", {}).get("learner_transfer_authorized"):
        raise ValueError("joint transfer requires a qualified target")
    target_rows = target["rows"]["train"]
    fit_games, holdout_games = _split_games(
        target_rows, fraction=config.holdout_fraction, seed=config.split_seed
    )
    fit_rows = tuple(row for row in target_rows if str(row["game_id"]) in fit_games)
    holdout_rows = tuple(row for row in target_rows if str(row["game_id"]) in holdout_games)

    network_config = _network_config(run["config"])
    baseline_path = Path(run["baseline"]["path"])
    baseline = HarbiChessNetwork(network_config)
    baseline.load_weights(str(baseline_path))
    rules = PythonChessRules()
    all_records = read_shard(config.train_shard, rules=rules).records
    fit = _prepare_data(
        all_records,
        fit_rows,
        dataset["rows"]["train"],
        baseline,
        explicit_targets=True,
    )
    holdout = _prepare_data(
        all_records,
        holdout_rows,
        dataset["rows"]["train"],
        baseline,
        explicit_targets=True,
    )
    anchor_candidates = tuple(record for record in all_records if record.game_id in fit_games)
    anchor_records = tuple(
        sorted(
            anchor_candidates,
            key=lambda record: _hash_key(
                config.split_seed,
                f"{record.game_id}:{record.game_index}:{record.ply}",
            ),
        )[: config.anchor_positions]
    )
    if len(anchor_records) != config.anchor_positions:
        raise ValueError("insufficient unique broad replay anchor positions")
    anchor = _anchor_data(anchor_records, baseline)
    baseline_quality = _policy_quality(
        holdout.base_logits,
        holdout,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.split_seed,
    )
    entropy = _target_entropy(holdout.targets)
    baseline_ce = float(baseline_quality["uncertainty_policy_cross_entropy"])
    reducible_gap = baseline_ce - entropy
    if reducible_gap <= 0:
        raise ValueError("holdout target has no reducible KL gap")

    store = SnapshotStore(config.telemetry_path)
    dashboard = store.read()
    started = time.perf_counter()
    arms = []
    checkpoints: list[tuple[tuple[str, mx.array], ...]] = []
    for arm_index, policy_anchor_weight in enumerate(config.policy_anchor_weights):
        dashboard = replace(
            dashboard,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.TRAINING,
            mode_detail=f"KOK anchor {policy_anchor_weight:g} · 0/{config.steps}",
            pilot_status=PilotStatus.TRAINING,
            pilot_steps_planned=config.steps * len(config.policy_anchor_weights),
            pilot_steps_completed=arm_index * config.steps,
        )
        store.write_atomic(dashboard)
        mx.random.seed(config.arm_seed)
        network = HarbiChessNetwork(network_config)
        network.load_weights(str(baseline_path))
        learner = JointPolicyLearner(
            network,
            learning_rate=config.learning_rate,
            policy_anchor_weight=policy_anchor_weight,
            wdl_anchor_weight=config.wdl_anchor_weight,
            max_gradient_norm=config.max_gradient_norm,
        )
        target_sampler = GameBalancedSampler(fit.records, seed=config.arm_seed)
        anchor_sampler = GameBalancedSampler(anchor.records, seed=config.arm_seed + 100)
        maximum_gradient_norm = 0.0
        for _step in range(config.steps):
            target_indices = target_sampler.sample_indices(config.target_batch_size)
            target_arrays = fit.select(target_indices)
            anchor_arrays = anchor.select(anchor_sampler.sample_indices(config.anchor_batch_size))
            arrays = (
                target_arrays[0],
                target_arrays[2],
                target_arrays[3],
                *anchor_arrays,
            )
            _, norm = learner.train_step(arrays)
            maximum_gradient_norm = max(maximum_gradient_norm, norm)
        holdout_policy, _ = network(holdout.inputs)
        quality = _policy_quality(
            holdout_policy,
            holdout,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.arm_seed,
        )
        gap_fraction = (
            baseline_ce - float(quality["uncertainty_policy_cross_entropy"])
        ) / reducible_gap
        policy_kl, wdl_kl, score_drift = _anchor_drift(network, anchor)
        reasons = _joint_reasons(
            quality,
            gap_fraction=gap_fraction,
            policy_kl=policy_kl,
            wdl_kl=wdl_kl,
            expected_score_drift=score_drift,
            maximum_gradient_norm=maximum_gradient_norm,
            config=config,
        )
        arms.append(
            {
                "policy_anchor_weight": policy_anchor_weight,
                "seed": config.arm_seed,
                "quality": quality,
                "reducible_gap_fraction": gap_fraction,
                "policy_anchor_kl": policy_kl,
                "wdl_anchor_kl": wdl_kl,
                "expected_score_drift": score_drift,
                "maximum_gradient_norm": maximum_gradient_norm,
                "passed": not reasons,
                "reasons": reasons,
            }
        )
        checkpoint = tuple(
            (name, mx.array(value)) for name, value in tree_flatten(network.parameters())
        )
        mx.eval([value for _, value in checkpoint])
        checkpoints.append(checkpoint)

    eligible = [index for index, row in enumerate(arms) if row["passed"]]
    selected_index = (
        min(
            eligible,
            key=lambda index: (
                float(arms[index]["quality"]["uncertainty_policy_cross_entropy"]),
                -float(arms[index]["policy_anchor_weight"]),
            ),
        )
        if eligible
        else None
    )
    selected = None
    if selected_index is not None:
        network = HarbiChessNetwork(network_config)
        network.load_weights(list(checkpoints[selected_index]))
        mx.eval(network.parameters())
        checkpoint_dir = config.output_dir / "candidate"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "model.safetensors"
        temporary = checkpoint_dir / ".model.tmp.safetensors"
        network.save_weights(str(temporary))
        os.replace(temporary, checkpoint_path)
        selected = {
            "arm_index": selected_index,
            "policy_anchor_weight": arms[selected_index]["policy_anchor_weight"],
            "path": str(checkpoint_path),
            "model_sha256": _sha256(checkpoint_path),
            "external_validation_authorized": True,
            "search_qualification_authorized": False,
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
                        "policy_target_result",
                        "dataset_result",
                        "run_result",
                        "train_shard",
                        "output_dir",
                        "telemetry_path",
                    )
                },
            },
            "split": {
                "fit_games": sorted(fit_games),
                "holdout_games": sorted(holdout_games),
                "fit_positions": len(fit.records),
                "holdout_positions": len(holdout.records),
                "anchor_positions": len(anchor.records),
            },
            "holdout_baseline_quality": baseline_quality,
            "holdout_target_entropy": entropy,
            "holdout_reducible_kl_gap": reducible_gap,
            "arms": arms,
            "elapsed_seconds": time.perf_counter() - started,
            "passed": selected is not None,
            "checkpoint": selected,
            "external_validation_authorized": selected is not None,
            "search_qualification_authorized": False,
            "arena_authorized": False,
            "generation_authorized": False,
            "promotion_authorized": False,
        },
    )
    dashboard = replace(
        dashboard,
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.IDLE,
        mode_detail=(
            f"KOK passed · anchor {selected['policy_anchor_weight']:g}"
            if selected
            else "KOK failed · external validation blocked"
        ),
        pilot_status=PilotStatus.PASSED if selected else PilotStatus.FAILED,
        pilot_steps_attempted=config.steps * len(config.policy_anchor_weights),
        pilot_steps_completed=config.steps * len(config.policy_anchor_weights),
        pilot_stop_reason="fixed_joint_arm_matrix",
        pilot_stop_detail=(
            "External validation authorized"
            if selected
            else "No joint representation arm passed internal holdout"
        ),
        promotion_ready=False,
    )
    store.write_atomic(dashboard)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-target-result", required=True, type=Path)
    parser.add_argument("--dataset-result", required=True, type=Path)
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--train-shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    arguments = parser.parse_args(argv)
    result = run_joint_policy_transfer(
        JointPolicyTransferConfig(
            policy_target_result=arguments.policy_target_result,
            dataset_result=arguments.dataset_result,
            run_result=arguments.run_result,
            train_shard=arguments.train_shard,
            output_dir=arguments.output_dir,
            telemetry_path=arguments.telemetry,
        )
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
