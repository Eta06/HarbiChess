"""Test broad replay policy anchoring for transferable SIPER learning."""

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
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.learner import MLXLearner
from harbichess.training.policy_projection import _target_entropy
from harbichess.training.uncertainty_policy_transfer import (
    LowRankPolicyAdapter,
    _clone_adapter,
    _merged_network,
    _network_config,
    _policy_quality,
    _prepare_data,
    _snapshot,
)


@dataclass(frozen=True, slots=True)
class AnchoredPolicyTransferConfig:
    policy_target_result: Path
    dataset_result: Path
    run_result: Path
    train_shard: Path
    output_dir: Path
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    holdout_fraction: float = 0.20
    anchor_positions: int = 2_048
    rank: int = 32
    steps: int = 960
    target_batch_size: int = 16
    anchor_batch_size: int = 64
    learning_rate: float = 1e-3
    anchor_weights: tuple[float, ...] = (0.25, 1.0, 4.0)
    split_seed: int = 2026082852
    arm_seeds: tuple[int, ...] = (2026082853, 2026082854, 2026082855)
    max_gradient_norm: float = 5.0
    bootstrap_samples: int = 2_000
    minimum_gap_fraction: float = 0.20
    minimum_teacher_spearman: float = 0.35
    maximum_harmful_ratio: float = 0.10
    maximum_verified_regret: float = 0.10
    minimum_best_action_coverage: float = 0.80
    maximum_anchor_kl: float = 0.02

    def __post_init__(self) -> None:
        if (
            min(
                self.anchor_positions,
                self.rank,
                self.steps,
                self.target_batch_size,
                self.anchor_batch_size,
                self.split_seed,
                self.bootstrap_samples,
            )
            <= 0
            or not 0 < self.holdout_fraction < 1
            or self.learning_rate <= 0
            or len(self.anchor_weights) != len(self.arm_seeds)
            or any(value <= 0 for value in (*self.anchor_weights, *self.arm_seeds))
            or self.max_gradient_norm <= 0
            or self.maximum_anchor_kl < 0
        ):
            raise ValueError("anchored policy transfer configuration is invalid")


@dataclass(frozen=True, slots=True)
class AnchorData:
    records: tuple[ReplayRecord, ...]
    features: mx.array
    base_logits: mx.array
    legal_masks: mx.array

    def select(self, indices: tuple[int, ...]) -> tuple[mx.array, ...]:
        rows = mx.array(indices, dtype=mx.int32)
        return tuple(
            mx.take(array, rows, axis=0)
            for array in (self.features, self.base_logits, self.legal_masks)
        )


class AnchoredPolicyLearner:
    def __init__(
        self,
        adapter: LowRankPolicyAdapter,
        *,
        learning_rate: float,
        anchor_weight: float,
        max_gradient_norm: float,
    ) -> None:
        self.adapter = adapter
        self.anchor_weight = anchor_weight
        self.max_gradient_norm = max_gradient_norm
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.0)
        self._loss_and_grad = nn.value_and_grad(adapter, self._loss)

    def _loss(
        self,
        target_features: mx.array,
        target_base_logits: mx.array,
        targets: mx.array,
        target_masks: mx.array,
        anchor_features: mx.array,
        anchor_base_logits: mx.array,
        anchor_masks: mx.array,
    ) -> mx.array:
        target_logits = mx.where(
            target_masks,
            self.adapter(target_features, target_base_logits),
            mx.array(-1e9),
        )
        target_loss = nn.losses.cross_entropy(target_logits, targets, reduction="mean")
        masked_base = mx.where(anchor_masks, anchor_base_logits, mx.array(-1e9))
        masked_candidate = mx.where(
            anchor_masks,
            self.adapter(anchor_features, anchor_base_logits),
            mx.array(-1e9),
        )
        base_log_probs = masked_base - mx.logsumexp(masked_base, axis=1, keepdims=True)
        candidate_log_probs = masked_candidate - mx.logsumexp(
            masked_candidate, axis=1, keepdims=True
        )
        base_probs = mx.exp(base_log_probs)
        anchor_kl = mx.mean(mx.sum(base_probs * (base_log_probs - candidate_log_probs), axis=1))
        return target_loss + self.anchor_weight * anchor_kl

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
            raise RuntimeError("anchored policy loss or gradients became non-finite")
        self.optimizer.update(self.adapter, gradients)
        mx.eval(self.adapter.parameters(), self.optimizer.state)
        return loss_value, norm_value


def _hash_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _split_games(
    rows: Sequence[Mapping[str, object]], *, fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    games = sorted({str(row["game_id"]) for row in rows}, key=lambda game: _hash_key(seed, game))
    holdout_count = max(1, round(len(games) * fraction))
    holdout = set(games[:holdout_count])
    return set(games[holdout_count:]), holdout


def _anchor_data(records: tuple[ReplayRecord, ...], base: HarbiChessNetwork) -> AnchorData:
    batch = MLXLearner.prepare_batch(build_training_batch(records))
    trunk = base._trunk(batch.inputs)
    features = mx.stop_gradient(base._policy_features(trunk))
    logits = mx.stop_gradient(base.policy_linear(features))
    mx.eval(features, logits, batch.legal_masks)
    return AnchorData(records, features, logits, batch.legal_masks)


def _anchor_kl(adapter: LowRankPolicyAdapter, anchor: AnchorData) -> float:
    base = mx.where(anchor.legal_masks, anchor.base_logits, mx.array(-1e9))
    candidate = mx.where(
        anchor.legal_masks,
        adapter(anchor.features, anchor.base_logits),
        mx.array(-1e9),
    )
    base_log = base - mx.logsumexp(base, axis=1, keepdims=True)
    candidate_log = candidate - mx.logsumexp(candidate, axis=1, keepdims=True)
    value = mx.mean(mx.sum(mx.exp(base_log) * (base_log - candidate_log), axis=1))
    mx.eval(value)
    return float(value.item())


def _reasons(
    quality: Mapping[str, object],
    *,
    gap_fraction: float,
    anchor_kl: float,
    maximum_gradient_norm: float,
    config: AnchoredPolicyTransferConfig,
) -> tuple[str, ...]:
    reasons = []
    if gap_fraction < config.minimum_gap_fraction:
        reasons.append("holdout reducible-gap closure is below 20%")
    if float(quality["mean_teacher_policy_spearman"]) < config.minimum_teacher_spearman:
        reasons.append("holdout teacher-policy Spearman is below 0.35")
    if float(quality["verified_delta_95_interval"][0]) <= 0:
        reasons.append("holdout verified-improvement interval is not positive")
    if float(quality["harmful_ratio"]) > config.maximum_harmful_ratio:
        reasons.append("holdout harmful-action ratio exceeds 10%")
    if float(quality["mean_verified_regret"]) > config.maximum_verified_regret:
        reasons.append("holdout mean verified regret exceeds 0.10")
    if float(quality["best_action_coverage_top_16"]) < config.minimum_best_action_coverage:
        reasons.append("holdout top-16 best-action coverage is below 80%")
    if anchor_kl > config.maximum_anchor_kl:
        reasons.append("broad replay anchor KL exceeds 0.02")
    if not math.isfinite(maximum_gradient_norm) or maximum_gradient_norm > (
        config.max_gradient_norm
    ):
        reasons.append("gradient safety limit was exceeded")
    return tuple(reasons)


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_anchored_policy_transfer(config: AnchoredPolicyTransferConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"anchored transfer output exists: {config.output_dir}")
    target = json.loads(config.policy_target_result.read_text(encoding="utf-8"))
    dataset = json.loads(config.dataset_result.read_text(encoding="utf-8"))
    run = json.loads(config.run_result.read_text(encoding="utf-8"))
    if not target.get("gate", {}).get("learner_transfer_authorized"):
        raise ValueError("anchored transfer requires a qualified target")
    target_rows = target["rows"]["train"]
    fit_games, holdout_games = _split_games(
        target_rows, fraction=config.holdout_fraction, seed=config.split_seed
    )
    fit_rows = tuple(row for row in target_rows if str(row["game_id"]) in fit_games)
    holdout_rows = tuple(row for row in target_rows if str(row["game_id"]) in holdout_games)

    network_config = _network_config(run["config"])
    baseline_path = Path(run["baseline"]["path"])
    base = HarbiChessNetwork(network_config)
    base.load_weights(str(baseline_path))
    rules = PythonChessRules()
    all_records = read_shard(config.train_shard, rules=rules).records
    fit = _prepare_data(
        all_records,
        fit_rows,
        dataset["rows"]["train"],
        base,
        explicit_targets=True,
    )
    holdout = _prepare_data(
        all_records,
        holdout_rows,
        dataset["rows"]["train"],
        base,
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
    anchor = _anchor_data(anchor_records, base)
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
    saved_weights = []
    for arm_index, (anchor_weight, seed) in enumerate(
        zip(config.anchor_weights, config.arm_seeds, strict=True)
    ):
        dashboard = replace(
            dashboard,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.TRAINING,
            mode_detail=f"CIPA anchor {anchor_weight:g} · 0/{config.steps}",
            pilot_status=PilotStatus.TRAINING,
            pilot_steps_planned=config.steps * len(config.anchor_weights),
            pilot_steps_completed=arm_index * config.steps,
        )
        store.write_atomic(dashboard)
        mx.random.seed(seed)
        adapter = LowRankPolicyAdapter(int(fit.features.shape[1]), config.rank)
        learner = AnchoredPolicyLearner(
            adapter,
            learning_rate=config.learning_rate,
            anchor_weight=anchor_weight,
            max_gradient_norm=config.max_gradient_norm,
        )
        target_sampler = GameBalancedSampler(fit.records, seed=seed)
        anchor_sampler = GameBalancedSampler(anchor.records, seed=seed + 100)
        maximum_gradient_norm = 0.0
        for _step in range(config.steps):
            arrays = (
                *fit.select(target_sampler.sample_indices(config.target_batch_size)),
                *anchor.select(anchor_sampler.sample_indices(config.anchor_batch_size)),
            )
            _, norm = learner.train_step(arrays)
            maximum_gradient_norm = max(maximum_gradient_norm, norm)
        quality = _policy_quality(
            adapter(holdout.features, holdout.base_logits),
            holdout,
            bootstrap_samples=config.bootstrap_samples,
            seed=seed,
        )
        gap_fraction = (
            baseline_ce - float(quality["uncertainty_policy_cross_entropy"])
        ) / reducible_gap
        anchor_kl = _anchor_kl(adapter, anchor)
        reasons = _reasons(
            quality,
            gap_fraction=gap_fraction,
            anchor_kl=anchor_kl,
            maximum_gradient_norm=maximum_gradient_norm,
            config=config,
        )
        arms.append(
            {
                "anchor_weight": anchor_weight,
                "seed": seed,
                "quality": quality,
                "reducible_gap_fraction": gap_fraction,
                "anchor_kl": anchor_kl,
                "maximum_gradient_norm": maximum_gradient_norm,
                "passed": not reasons,
                "reasons": reasons,
            }
        )
        saved_weights.append(_snapshot(adapter))

    eligible = [index for index, row in enumerate(arms) if row["passed"]]
    selected_index = (
        min(
            eligible,
            key=lambda index: (
                float(arms[index]["quality"]["uncertainty_policy_cross_entropy"]),
                -float(arms[index]["anchor_weight"]),
            ),
        )
        if eligible
        else None
    )
    checkpoint = None
    if selected_index is not None:
        selected_adapter = _clone_adapter(
            int(fit.features.shape[1]), config.rank, saved_weights[selected_index]
        )
        network = _merged_network(baseline_path, network_config, selected_adapter)
        checkpoint_dir = config.output_dir / "candidate"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_path = checkpoint_dir / "model.safetensors"
        temporary = checkpoint_dir / ".model.tmp.safetensors"
        network.save_weights(str(temporary))
        os.replace(temporary, checkpoint_path)
        checkpoint = {
            "arm_index": selected_index,
            "anchor_weight": arms[selected_index]["anchor_weight"],
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
            "passed": checkpoint is not None,
            "checkpoint": checkpoint,
            "external_validation_authorized": checkpoint is not None,
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
            f"CIPA passed · anchor {checkpoint['anchor_weight']:g}"
            if checkpoint
            else "CIPA failed · external validation blocked"
        ),
        pilot_status=PilotStatus.PASSED if checkpoint else PilotStatus.FAILED,
        pilot_steps_attempted=config.steps * len(config.anchor_weights),
        pilot_steps_completed=config.steps * len(config.anchor_weights),
        pilot_stop_reason="fixed_arm_matrix",
        pilot_stop_detail=(
            "External validation authorized"
            if checkpoint
            else "No replay-anchored arm passed internal holdout"
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
    result = run_anchored_policy_transfer(
        AnchoredPolicyTransferConfig(
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
