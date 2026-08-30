"""Run the frozen YELKEN stable-base/plastic-residual value ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.backends.plastic_value_network import (
    MIHVER_VALUE_PREFIXES,
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
from harbichess.replay.shard import read_shard
from harbichess.training.continuous_policy_iteration import _continuation_floor
from harbichess.training.decoupled_value_transfer import _material_quality, _MixedWDLSampler
from harbichess.training.full_gumbel_transfer import _network
from harbichess.training.invariant_wdl_transfer import _wdl_quality
from harbichess.training.joint_policy_value_transfer import (
    FixedOutcomeRatioGameBalancedSampler,
    _continuation_ranking,
    _parameter_hash,
    _sha256,
)


@dataclass(frozen=True, slots=True)
class StablePlasticAblationConfig:
    output_dir: Path
    value_result: Path
    model_path: Path
    source_continuous_result: Path
    runs_root: Path = Path("artifacts/runs")
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    expected_mihver_sha256: str = (
        "6c535585e952b3d8ba5ff2331fe4dc992d94c586fdfa6a7c3c0cbc9e2d229dbb"
    )
    steps: int = 40
    batch_size: int = 1024
    learning_rate: float = 1e-4
    ranking_positions: int = 32
    ranking_depth: int = 4
    seed: int = 2026083111
    tactical_seed: int = 2026082883
    search_workers: int = 24
    fixed_inference_batch_size: int = 12
    inference_wait_seconds: float = 0.00025

    def __post_init__(self) -> None:
        if (
            min(
                self.steps,
                self.batch_size,
                self.ranking_positions,
                self.ranking_depth,
                self.seed,
                self.tactical_seed,
                self.search_workers,
                self.fixed_inference_batch_size,
            )
            <= 0
            or self.batch_size % 8
            or self.learning_rate <= 0
            or self.inference_wait_seconds < 0
        ):
            raise ValueError("stable plastic ablation configuration is invalid")
        if len(self.expected_mihver_sha256) != 64:
            raise ValueError("expected MIHVER hash must be SHA-256")


@dataclass(frozen=True, slots=True)
class _Arm:
    name: str
    base_gradient_scale: float


_ARMS = (
    _Arm("frozen-base", 0.0),
    _Arm("low-lr-base", 0.1),
    _Arm("mutable-base-control", 1.0),
)


def _clone(network: HarbiChessPlasticValueNetwork) -> HarbiChessPlasticValueNetwork:
    clone = HarbiChessPlasticValueNetwork(
        network.config,
        invariant_config=network.invariant_config,
        plastic_config=network.plastic_config,
    )
    clone.load_weights(list(tree_flatten(network.parameters())))
    mx.eval(clone.parameters())
    return clone


def _load_mihver(config: StablePlasticAblationConfig) -> HarbiChessDecoupledValueNetwork:
    source = json.loads(config.value_result.read_text(encoding="utf-8"))
    selected = source.get("selected_wdl_arm")
    if not source.get("passed") or selected != "global-wdl":
        raise ValueError("YELKEN requires the qualified MIHVER global WDL arm")
    path = Path(source["wdl_arms"][selected]["model_path"])
    if _sha256(path) != config.expected_mihver_sha256:
        raise ValueError("MIHVER checkpoint hash does not match preregistration")
    network = HarbiChessDecoupledValueNetwork.from_base(_network())
    network.load_weights(str(path))
    mx.eval(network.parameters())
    return network


def _fresh_split(
    records: Sequence[ReplayRecord], *, seed: int
) -> tuple[tuple[ReplayRecord, ...], tuple[ReplayRecord, ...], dict[str, object]]:
    by_game: dict[str, list[ReplayRecord]] = defaultdict(list)
    for record in records:
        if record.outcome_value is not None:
            by_game[record.game_id].append(record)
    categories: dict[str, list[str]] = defaultdict(list)
    for game_id, rows in by_game.items():
        outcomes = {int(row.outcome_value) for row in rows if row.outcome_value is not None}
        category = "draw" if outcomes == {0} else "decisive"
        categories[category].append(game_id)

    def key(game_id: str) -> bytes:
        return hashlib.blake2b(f"{seed}:{game_id}".encode(), digest_size=16).digest()

    validation_games: set[str] = set()
    for game_ids in categories.values():
        ordered = sorted(game_ids, key=key)
        count = max(1, len(ordered) // 4)
        validation_games.update(ordered[:count])
    train = tuple(
        row
        for game_id, rows in sorted(by_game.items())
        if game_id not in validation_games
        for row in rows
    )
    validation = tuple(
        row
        for game_id, rows in sorted(by_game.items())
        if game_id in validation_games
        for row in rows
    )
    if {int(row.outcome_value) for row in train} != {-1, 0, 1}:
        raise ValueError("fresh fit split lacks a WDL outcome")
    return train, validation, {
        "games": len(by_game),
        "train_games": len(by_game) - len(validation_games),
        "validation_games": len(validation_games),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "overlap": bool((set(by_game) - validation_games) & validation_games),
    }


def _load_fresh_records(config: StablePlasticAblationConfig):
    result = json.loads(config.source_continuous_result.read_text(encoding="utf-8"))
    if not result.get("updates") or not all(row.get("accepted") for row in result["updates"]):
        raise ValueError("cached ablation requires the three accepted DEVRIYE updates")
    records = []
    paths = []
    for update in result["updates"]:
        path = Path(update["replay_path"])
        paths.append(str(path))
        records.extend(read_shard(path).records)
    return tuple(records), tuple(paths)


def _strict_wdl_reasons(
    baseline: dict[str, object], candidate: dict[str, object], *, label: str
) -> tuple[str, ...]:
    reasons = []
    for metric in ("cross_entropy", "macro_cross_entropy"):
        if float(candidate[metric]) > float(baseline[metric]):
            reasons.append(f"{label} {metric} regressed")
    if float(candidate["expected_score_pearson"]) < float(
        baseline["expected_score_pearson"]
    ):
        reasons.append(f"{label} expected-score Pearson regressed")
    if min(
        float(candidate["loss_draw_margin"]), float(candidate["win_draw_margin"])
    ) < 0.03:
        reasons.append(f"{label} outcome margin fell below 0.03")
    if float(candidate["ece_10"]) > 0.12:
        reasons.append(f"{label} ECE-10 exceeds 0.12")
    return tuple(reasons)


class _ValueLearner:
    def __init__(
        self,
        network: HarbiChessPlasticValueNetwork,
        *,
        learning_rate: float,
        base_gradient_scale: float,
    ) -> None:
        self.network = network
        self.base_gradient_scale = base_gradient_scale
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.loss_and_grad = nn.value_and_grad(network, self._loss)

    @staticmethod
    def _loss(network, inputs: mx.array, targets: mx.array) -> mx.array:
        return nn.losses.cross_entropy(network(inputs)[1], targets, reduction="mean")

    def step(self, inputs: mx.array, targets: mx.array) -> tuple[float, float]:
        loss, gradients = self.loss_and_grad(self.network, inputs, targets)
        gradients, norm = optim.clip_grad_norm(gradients, 5.0)
        if self.base_gradient_scale != 1.0:
            gradients = tree_unflatten(
                [
                    (
                        name,
                        value * self.base_gradient_scale
                        if name.startswith(MIHVER_VALUE_PREFIXES)
                        else value,
                    )
                    for name, value in tree_flatten(gradients)
                ]
            )
        finite = mx.all(
            mx.stack([mx.all(mx.isfinite(value)) for _, value in tree_flatten(gradients)])
        )
        mx.eval(loss, norm, finite, gradients)
        values = float(loss.item()), float(norm.item())
        if not bool(finite.item()) or not all(math.isfinite(value) for value in values):
            raise RuntimeError("plastic value learner produced non-finite values")
        self.optimizer.update(self.network, gradients)
        mx.eval(self.network.parameters(), self.optimizer.state)
        return values


def _save(network, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.safetensors")
    network.save_weights(str(temporary))
    os.replace(temporary, path)
    return _sha256(path)


def run_stable_plastic_ablation(config: StablePlasticAblationConfig) -> Path:
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
    historical_train_labels = mx.array(
        [{1: 0, 0: 1, -1: 2}[int(row.outcome_value)] for row in historical_train],
        dtype=mx.int32,
    )
    fresh_train_labels = mx.array(
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
    baseline_policy, baseline_value = mihver(historical_validation_inputs[:64])
    wrapped_policy, wrapped_value = baseline(historical_validation_inputs[:64])
    mx.eval(baseline_policy, baseline_value, wrapped_policy, wrapped_value)
    initial_delta = {
        "policy_max_abs": float(mx.max(mx.abs(baseline_policy - wrapped_policy)).item()),
        "value_max_abs": float(mx.max(mx.abs(baseline_value - wrapped_value)).item()),
    }
    if max(initial_delta.values()) > 1e-6:
        raise RuntimeError("plastic wrapper is not function preserving")

    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="YELKEN stable/plastic cached ablation",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=len(_ARMS) * config.steps,
        pilot_steps_completed=0,
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    arms = {}
    selected_arm = None
    completed_steps = 0
    for arm in _ARMS:
        mx.random.seed(config.seed)
        network = HarbiChessPlasticValueNetwork.from_mihver(mihver)
        if arm.base_gradient_scale == 0.0:
            network.freeze_to_plastic_value()
        elif arm.base_gradient_scale == 0.1:
            network.freeze_to_low_lr_continuous_heads()
            network.policy_conv.freeze()
            network.policy_linear.freeze()
        else:
            network.freeze_to_mutable_continuous_heads()
            network.policy_conv.freeze()
            network.policy_linear.freeze()
        trainable_names = tuple(name for name, _ in tree_flatten(network.trainable_parameters()))
        learner = _ValueLearner(
            network,
            learning_rate=config.learning_rate,
            base_gradient_scale=arm.base_gradient_scale,
        )
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
        initial_nontrainable = _parameter_hash(
            network, excluded_prefixes=(*PLASTIC_VALUE_PREFIXES, *MIHVER_VALUE_PREFIXES)
        )
        initial_stable_base = _parameter_hash(
            network, excluded_prefixes=PLASTIC_VALUE_PREFIXES
        )
        curve = []
        accepted = None
        for step in range(1, config.steps + 1):
            half = config.batch_size // 2
            historical_rows = mx.array(historical_sampler.sample_indices(half), dtype=mx.int32)
            fresh_rows = mx.array(fresh_sampler.sample_indices(half), dtype=mx.int32)
            inputs = mx.concatenate(
                (
                    mx.take(historical_train_inputs, historical_rows, axis=0),
                    mx.take(fresh_train_inputs, fresh_rows, axis=0),
                ),
                axis=0,
            )
            targets = mx.concatenate(
                (
                    mx.take(historical_train_labels, historical_rows, axis=0),
                    mx.take(fresh_train_labels, fresh_rows, axis=0),
                ),
                axis=0,
            )
            loss, gradient_norm = learner.step(inputs, targets)
            old_quality = _wdl_quality(network, historical_validation_inputs, historical_outcomes)
            fresh_quality = _wdl_quality(network, fresh_validation_inputs, fresh_outcomes)
            numeric_reasons = (
                *_strict_wdl_reasons(baseline_old, old_quality, label="old"),
                *_strict_wdl_reasons(baseline_fresh, fresh_quality, label="fresh"),
            )
            row = {
                "step": step,
                "loss": loss,
                "gradient_norm": gradient_norm,
                "old_wdl": old_quality,
                "fresh_wdl": fresh_quality,
                "numeric_reasons": numeric_reasons,
            }
            curve.append(row)
            completed_steps += 1
            snapshot = replace(
                snapshot,
                updated_at=datetime.now(UTC).isoformat(),
                mode_detail=f"YELKEN {arm.name} · step {step}/{config.steps}",
                pilot_steps_completed=completed_steps,
                training_step=completed_steps,
                value_loss=loss,
                total_loss=loss,
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
            model_path = config.output_dir / "arms" / arm.name / "model.safetensors"
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
        frozen_unchanged = _parameter_hash(
            network, excluded_prefixes=(*PLASTIC_VALUE_PREFIXES, *MIHVER_VALUE_PREFIXES)
        ) == initial_nontrainable
        stable_base_unchanged = _parameter_hash(
            network, excluded_prefixes=PLASTIC_VALUE_PREFIXES
        ) == initial_stable_base
        arms[arm.name] = {
            "base_gradient_scale": arm.base_gradient_scale,
            "trainable_parameters": trainable_names,
            "frozen_parameters_unchanged": frozen_unchanged,
            "stable_base_unchanged": stable_base_unchanged,
            "passed": (
                accepted is not None
                and frozen_unchanged
                and (arm.base_gradient_scale != 0.0 or stable_base_unchanged)
            ),
            "selected": accepted,
            "curve": curve,
        }
        if arms[arm.name]["passed"] and selected_arm is None:
            selected_arm = arm.name

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
            "initial_function_delta": initial_delta,
            "baseline": {
                "old_wdl": baseline_old,
                "fresh_wdl": baseline_fresh,
                "continuation": baseline_continuation,
                "tactical": baseline_tactical,
                "material": _material_quality(
                    baseline, *_prepare(historical_validation[:4096], rules)
                ),
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
        f"YELKEN cached ablation passed · {selected_arm} selected"
        if passed
        else "YELKEN cached ablation failed · fresh generation blocked"
    )
    store.write_atomic(
        replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.IDLE,
            mode_detail=detail,
            pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
            pilot_stop_reason="cached_ablation_gate",
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
    result = run_stable_plastic_ablation(
        StablePlasticAblationConfig(
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
