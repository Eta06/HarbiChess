"""Run a frozen rolling-replay/latest-network continuous learner pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import Side
from harbichess.dashboard.state import (
    CheckpointStatus,
    HistoryPoint,
    PilotStatus,
    RunMode,
    SnapshotStore,
)
from harbichess.evaluation.arena import _openings
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
from harbichess.evaluation.deterministic_value_probe import (
    _prepare as _prepare_value,
)
from harbichess.evaluation.deterministic_value_probe import _round_robin
from harbichess.evaluation.full_gumbel_targets import (
    _identity,
    _selection_summary,
    _target_row,
)
from harbichess.evaluation.full_gumbel_targets import (
    select_stratified_records as select_target_records,
)
from harbichess.evaluation.system_teacher_qualification import (
    _play_game,
    summarize_games,
)
from harbichess.evaluation.teacher_qualification import (
    _atomic_json,
    select_stratified_records,
)
from harbichess.replay.shard import ShardMetadata, write_shard_atomic
from harbichess.replay.split import ReplaySplit
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS
from harbichess.selfplay.continuous_replay import (
    ContinuousReplayConfig,
    generate_continuous_replay,
)
from harbichess.training.decoupled_value_transfer import (
    _material_quality,
    _MixedWDLSampler,
)
from harbichess.training.full_gumbel_transfer import (
    PreparedTransfer,
    _evaluator,
    _network,
    _policy_quality,
    _prepare,
)
from harbichess.training.invariant_wdl_transfer import _wdl_quality
from harbichess.training.joint_policy_value_transfer import (
    FixedOutcomeRatioGameBalancedSampler,
    _continuation_ranking,
    _parameter_hash,
    _sha256,
)


@dataclass(frozen=True, slots=True)
class ContinuousPolicyIterationConfig:
    output_dir: Path
    value_result: Path
    model_path: Path
    runs_root: Path = Path("artifacts/runs")
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    expected_initial_sha256: str = (
        "6c535585e952b3d8ba5ff2331fe4dc992d94c586fdfa6a7c3c0cbc9e2d229dbb"
    )
    updates: int = 3
    train_targets_per_update: int = 768
    validation_targets_per_update: int = 192
    rolling_generations: int = 2
    simulations: int = 256
    max_considered_actions: int = 16
    target_workers: int = 24
    fixed_inference_batch_size: int = 12
    inference_wait_seconds: float = 0.00025
    steps_per_update: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-4
    selfplay_games_per_update: int = 96
    selfplay_workers: int = 24
    selfplay_simulations: int = 64
    selfplay_max_plies: int = 96
    minimum_known_selfplay_games: int = 24
    ranking_positions: int = 32
    ranking_depth: int = 4
    tactical_seed: int = 2026082883
    arena_pairs_per_update: int = 4
    arena_simulations_per_update: int = 32
    final_arena_pairs: int = 8
    final_arena_simulations: int = 64
    arena_opening_plies: int = 8
    arena_max_plies: int = 96
    arena_workers: int = 16
    bootstrap_samples: int = 10_000
    seed: int = 2026083101

    def __post_init__(self) -> None:
        counts = (
            self.updates,
            self.train_targets_per_update,
            self.validation_targets_per_update,
            self.rolling_generations,
            self.simulations,
            self.max_considered_actions,
            self.target_workers,
            self.fixed_inference_batch_size,
            self.steps_per_update,
            self.batch_size,
            self.selfplay_games_per_update,
            self.selfplay_workers,
            self.selfplay_simulations,
            self.selfplay_max_plies,
            self.minimum_known_selfplay_games,
            self.ranking_positions,
            self.ranking_depth,
            self.tactical_seed,
            self.arena_pairs_per_update,
            self.arena_simulations_per_update,
            self.final_arena_pairs,
            self.final_arena_simulations,
            self.arena_max_plies,
            self.arena_workers,
            self.bootstrap_samples,
            self.seed,
        )
        if (
            min(counts) <= 0
            or self.learning_rate <= 0
            or self.inference_wait_seconds < 0
            or self.arena_opening_plies < 0
        ):
            raise ValueError("continuous policy iteration configuration is invalid")
        if self.rolling_generations > self.updates:
            raise ValueError("rolling generations cannot exceed update count")
        if self.batch_size % 2:
            raise ValueError("continuous value batch size must be even")
        if self.selfplay_games_per_update % 3:
            raise ValueError("self-play games must divide evenly across three phases")
        if self.minimum_known_selfplay_games > self.selfplay_games_per_update:
            raise ValueError("minimum known games cannot exceed self-play games")
        if len(self.expected_initial_sha256) != 64:
            raise ValueError("initial candidate hash must be SHA-256")


_TRAINABLE_PREFIXES = (
    "policy_conv.",
    "policy_linear.",
    "invariant_value_linear.",
    "global_value_hidden.",
    "global_value_output.",
)
_POLICY_PREFIXES = _TRAINABLE_PREFIXES[:2]
_VALUE_PREFIXES = _TRAINABLE_PREFIXES[2:]


@dataclass(frozen=True, slots=True)
class _LearnerState:
    step: int
    weights: tuple[tuple[str, mx.array], ...]
    optimizer: tuple[tuple[str, mx.array], ...]


class _ContinuousHeadLearner:
    def __init__(self, network: HarbiChessDecoupledValueNetwork, *, learning_rate: float) -> None:
        self.network = network
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.step = 0
        self._loss_and_grad = nn.value_and_grad(network, self._loss)

    @staticmethod
    def _loss(
        network,
        policy_inputs: mx.array,
        policy_targets: mx.array,
        legal_masks: mx.array,
        value_inputs: mx.array,
        value_targets: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        policy_logits = network(policy_inputs)[0]
        value_logits = network(value_inputs)[1]
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
    ) -> dict[str, float | int]:
        (total, policy, value), gradients = self._loss_and_grad(
            self.network,
            policy_inputs,
            policy_targets,
            legal_masks,
            value_inputs,
            value_targets,
        )
        gradients, norm = optim.clip_grad_norm(gradients, 5.0)
        mx.eval(total, policy, value, norm, gradients)
        values = tuple(float(item.item()) for item in (total, policy, value, norm))
        if not all(math.isfinite(item) for item in values):
            raise RuntimeError("continuous learner produced non-finite loss or gradients")
        self.optimizer.update(self.network, gradients)
        mx.eval(self.network.parameters(), self.optimizer.state)
        self.step += 1
        return {
            "step": self.step,
            "total_loss": values[0],
            "policy_loss": values[1],
            "value_loss": values[2],
            "gradient_norm": values[3],
        }

    def snapshot(self) -> _LearnerState:
        weights = tuple(
            (name, mx.array(value)) for name, value in tree_flatten(self.network.parameters())
        )
        optimizer = tuple(
            (name, mx.array(value)) for name, value in tree_flatten(self.optimizer.state)
        )
        mx.eval(
            [value for _, value in weights],
            [value for _, value in optimizer],
        )
        return _LearnerState(self.step, weights, optimizer)

    def restore(self, state: _LearnerState) -> None:
        self.network.load_weights(list(state.weights))
        self.optimizer.state = tree_unflatten(list(state.optimizer))
        self.step = state.step
        mx.eval(self.network.parameters(), self.optimizer.state)

    def reset_optimizer(self) -> None:
        self.optimizer = optim.Adam(learning_rate=self.learning_rate)


def _clone(network) -> HarbiChessDecoupledValueNetwork:
    clone = HarbiChessDecoupledValueNetwork.from_base(_network())
    clone.load_weights(list(tree_flatten(network.parameters())))
    mx.eval(clone.parameters())
    return clone


def _save_network(network, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.safetensors")
    network.save_weights(str(temporary))
    os.replace(temporary, path)
    return _sha256(path)


def _combine_policy(generations: Sequence[PreparedTransfer]) -> PreparedTransfer:
    if not generations:
        raise ValueError("rolling policy buffer requires target generations")
    combined = PreparedTransfer(
        tuple(record for generation in generations for record in generation.records),
        mx.concatenate(tuple(generation.inputs for generation in generations), axis=0),
        mx.concatenate(tuple(generation.targets for generation in generations), axis=0),
        mx.concatenate(tuple(generation.legal_masks for generation in generations), axis=0),
        tuple(outcome for generation in generations for outcome in generation.wdl_targets),
    )
    mx.eval(combined.inputs, combined.targets, combined.legal_masks)
    return combined


def _generate_targets(
    network,
    selected: dict[str, tuple],
    *,
    config: ContinuousPolicyIterationConfig,
    update: int,
) -> tuple[dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    rules = PythonChessRules()
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(network, fixed_batch_size=config.fixed_inference_batch_size),
        max_batch_size=min(config.fixed_inference_batch_size, config.target_workers),
        max_wait_seconds=config.inference_wait_seconds,
    )
    evaluator = NeuralPositionEvaluator(batcher, rules=rules)
    search = FullGumbelMCTS(
        evaluator,
        rules=rules,
        config=FullGumbelConfig(
            simulations=config.simulations,
            max_considered_actions=config.max_considered_actions,
            gumbel_scale=0.0,
        ),
    )
    started = time.perf_counter()
    rows = {}
    try:
        for partition, records in selected.items():

            def build(record):
                return _target_row(
                    record,
                    search,
                    seed=config.seed + update * 100,
                    rules=rules,
                )

            with ThreadPoolExecutor(max_workers=min(config.target_workers, len(records))) as pool:
                rows[partition] = tuple(pool.map(build, records))
    finally:
        batcher.close()
    elapsed = time.perf_counter() - started
    return rows, {
        "elapsed_seconds": elapsed,
        "inference": {
            **asdict(batcher.statistics),
            "positions_per_second": batcher.statistics.positions / max(elapsed, 1e-9),
        },
        "selection": {
            partition: _selection_summary(partition_rows)
            for partition, partition_rows in rows.items()
        },
    }


def _paired_arena(
    previous,
    candidate,
    *,
    pairs: int,
    simulations: int,
    seed: int,
    config: ContinuousPolicyIterationConfig,
) -> dict[str, object]:
    rules = PythonChessRules()
    previous_batcher, previous_evaluator = _evaluator(
        previous,
        config.arena_workers,
        config.fixed_inference_batch_size,
        config.inference_wait_seconds,
    )
    candidate_batcher, candidate_evaluator = _evaluator(
        candidate,
        config.arena_workers,
        config.fixed_inference_batch_size,
        config.inference_wait_seconds,
    )
    previous_search = FullGumbelMCTS(
        previous_evaluator,
        rules=rules,
        config=FullGumbelConfig(
            simulations=simulations,
            max_considered_actions=config.max_considered_actions,
            gumbel_scale=0.0,
        ),
    )
    candidate_search = FullGumbelMCTS(
        candidate_evaluator,
        rules=rules,
        config=FullGumbelConfig(
            simulations=simulations,
            max_considered_actions=config.max_considered_actions,
            gumbel_scale=0.0,
        ),
    )
    openings = _openings(
        rules,
        count=pairs,
        plies=config.arena_opening_plies,
        seed=seed,
    )
    tasks = tuple(
        (pair, side, state, moves)
        for pair, (state, moves) in enumerate(openings)
        for side in (Side.WHITE, Side.BLACK)
    )

    def play(task):
        pair, side, state, moves = task
        white = candidate_search if side is Side.WHITE else previous_search
        black = candidate_search if side is Side.BLACK else previous_search
        return _play_game(
            white,
            black,
            rules,
            state,
            pair_index=pair,
            candidate_side=side,
            opening_moves=moves,
            max_plies=config.arena_max_plies,
        )

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=config.arena_workers) as pool:
            games = tuple(pool.map(play, tasks))
    finally:
        previous_batcher.close()
        candidate_batcher.close()
    return {
        **summarize_games(
            games,
            bootstrap_samples=config.bootstrap_samples,
            seed=seed,
        ),
        "simulations": simulations,
        "opening_pairs": pairs,
        "elapsed_seconds": time.perf_counter() - started,
        "previous_inference": asdict(previous_batcher.statistics),
        "candidate_inference": asdict(candidate_batcher.statistics),
    }


def _policy_gate(before: dict[str, float], after: dict[str, float]) -> tuple[str, ...]:
    reasons = []
    if before["cross_entropy"] - after["cross_entropy"] < 0.01:
        reasons.append("fresh teacher policy CE improvement is below 0.01")
    if after["top_action_agreement"] < before["top_action_agreement"]:
        reasons.append("fresh teacher top-action agreement regressed")
    return tuple(reasons)


def _continuous_wdl_gate(
    previous: dict[str, object], candidate: dict[str, object]
) -> tuple[str, ...]:
    reasons = []
    if float(candidate["cross_entropy"]) > float(previous["cross_entropy"]) + 0.01:
        reasons.append("WDL micro CE regressed by more than 0.01")
    if float(candidate["macro_cross_entropy"]) > float(previous["macro_cross_entropy"]) + 0.01:
        reasons.append("WDL macro CE regressed by more than 0.01")
    if float(candidate["expected_score_pearson"]) < (
        float(previous["expected_score_pearson"]) - 0.02
    ):
        reasons.append("WDL expected-score Pearson regressed by more than 0.02")
    if float(candidate["cross_entropy"]) > 0.9996843744333518:
        reasons.append("WDL micro CE lost the MIHVER floor")
    if float(candidate["macro_cross_entropy"]) > 0.998904546185216:
        reasons.append("WDL macro CE lost the MIHVER floor")
    if float(candidate["expected_score_pearson"]) < 0.20:
        reasons.append("WDL Pearson lost the MIHVER floor")
    if (
        min(
            float(candidate["loss_draw_margin"]),
            float(candidate["win_draw_margin"]),
        )
        < 0.03
    ):
        reasons.append("WDL outcome margin lost the MIHVER floor")
    if float(candidate["ece_10"]) > 0.12:
        reasons.append("WDL ECE-10 exceeds 0.12")
    return tuple(reasons)


def _continuation_floor(result: dict[str, object]) -> tuple[str, ...]:
    reasons = []
    if float(result["candidate_mean_spearman"]) < 0.05:
        reasons.append("continuation mean Spearman is below 0.05")
    if float(result["candidate_verified_top_agreement"]) < 0.34375:
        reasons.append("continuation verified-top agreement is below 0.34375")
    return tuple(reasons)


def _select_numeric_checkpoint(checkpoints: Sequence[dict[str, object]]):
    if not checkpoints:
        raise ValueError("continuous update produced no validation checkpoints")
    eligible = [checkpoint for checkpoint in checkpoints if not checkpoint["reasons"]]
    selected = min(
        eligible or checkpoints,
        key=lambda checkpoint: (
            int(checkpoint["local_step"]),
            float(checkpoint["policy"]["cross_entropy"]),  # type: ignore[index]
            float(checkpoint["wdl"]["macro_cross_entropy"]),  # type: ignore[index]
        ),
    )
    return selected, bool(eligible)


def _compose_headwise_state(
    policy_checkpoint: dict[str, object],
    value_checkpoint: dict[str, object],
) -> _LearnerState:
    policy_state = policy_checkpoint["state"]
    value_state = value_checkpoint["state"]
    if not isinstance(policy_state, _LearnerState) or not isinstance(value_state, _LearnerState):
        raise TypeError("head-wise checkpoints require learner states")
    policy_weights = dict(policy_state.weights)
    value_weights = dict(value_state.weights)
    weights = tuple(
        (
            name,
            value_weights[name] if name.startswith(_VALUE_PREFIXES) else value,
        )
        for name, value in policy_state.weights
    )
    if set(policy_weights) != set(value_weights):
        raise ValueError("head-wise checkpoint parameter trees differ")
    return _LearnerState(
        step=max(policy_state.step, value_state.step),
        weights=weights,
        optimizer=policy_state.optimizer,
    )


def _select_continuation_starts(
    records,
    *,
    updates: int,
    games_per_update: int,
    seed: int,
):
    per_phase_per_update = games_per_update // 3
    required_per_phase = updates * per_phase_per_update
    phases = {
        "opening": tuple(record for record in records if record.ply < 20),
        "middlegame": tuple(record for record in records if 20 <= record.ply < 80),
        "endgame": tuple(record for record in records if record.ply >= 80),
    }

    def key(label: str) -> bytes:
        return hashlib.blake2b(f"{seed}:{label}".encode(), digest_size=16).digest()

    selected_by_phase = {}
    for phase, candidates in phases.items():
        by_game = {}
        for record in candidates:
            by_game.setdefault(record.game_id, []).append(record)
        chosen = []
        for game_id in sorted(by_game, key=lambda value: key(f"game:{phase}:{value}")):
            rows = sorted(by_game[game_id], key=lambda record: key(_identity(record)))
            chosen.append(rows[0])
            if len(chosen) == required_per_phase:
                break
        if len(chosen) < required_per_phase:
            raise ValueError(
                f"continuation pool has fewer than {required_per_phase} distinct {phase} games"
            )
        selected_by_phase[phase] = tuple(chosen)
    return tuple(
        tuple(
            selected_by_phase[phase][
                update * per_phase_per_update : (update + 1) * per_phase_per_update
            ][offset]
            for phase in ("opening", "middlegame", "endgame")
            for offset in range(per_phase_per_update)
        )
        for update in range(updates)
    )


def _load_initial(
    config: ContinuousPolicyIterationConfig,
) -> tuple[HarbiChessDecoupledValueNetwork, Path]:
    source = json.loads(config.value_result.read_text(encoding="utf-8"))
    selected = source.get("selected_wdl_arm")
    if not source.get("passed") or selected != "global-wdl":
        raise ValueError("continuous pilot requires the qualified MIHVER global WDL arm")
    path = Path(source["wdl_arms"][selected]["model_path"])
    if _sha256(path) != config.expected_initial_sha256:
        raise ValueError("MIHVER initial checkpoint hash does not match preregistration")
    network = HarbiChessDecoupledValueNetwork.from_base(_network())
    network.load_weights(str(path))
    network.freeze_to_continuous_heads()
    mx.eval(network.parameters())
    return network, path


def run_continuous_policy_iteration(config: ContinuousPolicyIterationConfig) -> Path:
    if config.output_dir.exists():
        raise FileExistsError(f"continuous pilot output exists: {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    network, initial_path = _load_initial(config)
    initial_network = _clone(network)
    release = _network()
    release.load_weights(str(config.model_path))
    learner = _ContinuousHeadLearner(network, learning_rate=config.learning_rate)
    nontrainable_hash = _parameter_hash(network, excluded_prefixes=_TRAINABLE_PREFIXES)

    pool_config = CorrectedReplayValueTransferConfig(
        output_dir=config.output_dir,
        model_path=config.model_path,
        runs_root=config.runs_root,
    )
    games, provenance = _load_games(pool_config)
    train_records, validation_records, split = _split_games(games, seed=pool_config.seed)
    rules = PythonChessRules()
    wdl_train_inputs, _ = _prepare_value(train_records, rules)
    wdl_validation_inputs, _ = _prepare_value(validation_records, rules)
    train_outcomes = tuple(int(record.outcome_value) for record in train_records)
    validation_outcomes = tuple(int(record.outcome_value) for record in validation_records)
    wdl_labels = mx.array([{1: 0, 0: 1, -1: 2}[value] for value in train_outcomes], dtype=mx.int32)
    material_records = _round_robin(validation_records, 4096)
    material_validation = _prepare_value(material_records, rules)
    material_baseline = _material_quality(network, *material_validation)
    ranking_records = select_stratified_records(
        validation_records,
        rules=rules,
        count=config.ranking_positions,
        seed=2026083091,
    )
    continuation_starts = _select_continuation_starts(
        train_records,
        updates=config.updates,
        games_per_update=config.selfplay_games_per_update,
        seed=config.seed + 3000,
    )

    store = SnapshotStore(config.telemetry_path)
    snapshot = replace(
        store.read(),
        updated_at=datetime.now(UTC).isoformat(),
        mode=RunMode.TRAINING,
        mode_detail="DEVRIYE continuous pilot · baseline qualification",
        run_id=config.output_dir.name,
        pilot_status=PilotStatus.TRAINING,
        pilot_steps_planned=config.updates * config.steps_per_update,
        pilot_steps_completed=0,
        promotion_ready=False,
    )
    store.write_atomic(snapshot)
    started = time.perf_counter()
    initial_wdl = _wdl_quality(network, wdl_validation_inputs, validation_outcomes)
    initial_continuation = _continuation_ranking(
        release,
        network,
        ranking_records,
        rules=rules,
        depth=config.ranking_depth,
    )
    tactical_config = replace(
        DecoupledValueQualificationConfig(
            output_dir=config.output_dir,
            value_result=config.value_result,
            model_path=config.model_path,
        ),
        tactical_seed=config.tactical_seed,
        search_workers=config.target_workers,
        fixed_inference_batch_size=config.fixed_inference_batch_size,
        inference_wait_seconds=config.inference_wait_seconds,
    )
    initial_tactical = _tactical(_clone(network), config=tactical_config)
    initial_checkpoint = config.output_dir / "checkpoints" / "update-000" / "model.safetensors"
    initial_sha256 = _save_network(network, initial_checkpoint)

    used_train: set[str] = set()
    used_validation: set[str] = set()
    rolling: list[PreparedTransfer] = []
    rolling_value_records = []
    accepted = []
    target_artifacts = []
    stopped_reason = None
    for update in range(1, config.updates + 1):
        before_state = learner.snapshot()
        previous_network = _clone(network)
        previous_wdl = _wdl_quality(previous_network, wdl_validation_inputs, validation_outcomes)
        available_train = tuple(
            record for record in train_records if _identity(record) not in used_train
        )
        available_validation = tuple(
            record for record in validation_records if _identity(record) not in used_validation
        )
        selected = {
            "train": select_target_records(
                available_train,
                count=config.train_targets_per_update,
                seed=config.seed + update * 2,
                rules=rules,
            ),
            "validation": select_target_records(
                available_validation,
                count=config.validation_targets_per_update,
                seed=config.seed + update * 2 + 1,
                rules=rules,
            ),
        }
        used_train.update(_identity(record) for record in selected["train"])
        used_validation.update(_identity(record) for record in selected["validation"])
        teacher_sha256 = _sha256(
            initial_checkpoint
            if update == 1
            else config.output_dir
            / "checkpoints"
            / f"update-{update - 1:03d}"
            / "model.safetensors"
        )
        completed_selfplay_games = 0

        def on_game_complete(_game, current_update=update) -> None:
            nonlocal completed_selfplay_games, snapshot
            completed_selfplay_games += 1
            elapsed = time.perf_counter() - started
            snapshot = replace(
                snapshot,
                updated_at=datetime.now(UTC).isoformat(),
                mode=RunMode.SELF_PLAY,
                mode_detail=(
                    f"DEVRIYE latest-network replay · update {current_update}/{config.updates} · "
                    f"{completed_selfplay_games}/{config.selfplay_games_per_update} games"
                ),
                active_games=config.selfplay_games_per_update - completed_selfplay_games,
                completed_games=completed_selfplay_games,
                lifetime_games=snapshot.lifetime_games + 1,
                games_per_hour=completed_selfplay_games / max(elapsed, 1e-9) * 3600.0,
            )
            store.write_atomic(snapshot)

        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.SELF_PLAY,
            mode_detail=f"DEVRIYE latest-network replay · update {update}/{config.updates}",
            active_games=config.selfplay_games_per_update,
            completed_games=0,
        )
        store.write_atomic(snapshot)
        replay_run_id = f"{config.output_dir.name}-update-{update:03d}"
        _, replay_records, replay_metrics = generate_continuous_replay(
            _clone(network),
            run_id=replay_run_id,
            run_seed=config.seed + update * 10_000,
            config=ContinuousReplayConfig(
                games=config.selfplay_games_per_update,
                workers=config.selfplay_workers,
                simulations=config.selfplay_simulations,
                max_considered_actions=config.max_considered_actions,
                max_plies=config.selfplay_max_plies,
                fixed_inference_batch_size=config.fixed_inference_batch_size,
                inference_wait_seconds=config.inference_wait_seconds,
            ),
            on_game_complete=on_game_complete,
            initial_states=tuple(record.state for record in continuation_starts[update - 1]),
        )
        replay_metrics["starting_records"] = tuple(
            _identity(record) for record in continuation_starts[update - 1]
        )
        replay_path = config.output_dir / "replay" / f"update-{update:03d}.jsonl.gz"
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        replay_header = write_shard_atomic(
            replay_path,
            replay_records,
            ShardMetadata(
                run_id=replay_run_id,
                generation=update,
                source_checkpoint=teacher_sha256,
                source_commit=source_commit,
                created_at=datetime.now(UTC).isoformat(),
                split=ReplaySplit.TRAIN,
            ),
        )
        known_replay_records = tuple(
            record for record in replay_records if record.outcome_value is not None
        )
        if int(replay_metrics["known_outcome_games"]) < config.minimum_known_selfplay_games:
            reason = (
                "fresh replay has fewer than the preregistered minimum of "
                f"{config.minimum_known_selfplay_games} known-outcome games"
            )
            accepted.append(
                {
                    "update": update,
                    "accepted": False,
                    "rollback": True,
                    "reasons": [reason],
                    "teacher_sha256": teacher_sha256,
                    "selfplay": replay_metrics,
                    "replay_path": str(replay_path),
                    "replay_header": asdict(replay_header),
                }
            )
            stopped_reason = f"update-{update:03d}-insufficient-fresh-value-replay"
            break
        rolling_value_records.append(known_replay_records)
        rolling_value_records = rolling_value_records[-config.rolling_generations :]
        fresh_value_records = tuple(
            record for generation in rolling_value_records for record in generation
        )
        fresh_outcomes = {record.outcome_value for record in fresh_value_records}
        if fresh_outcomes != {-1, 0, 1}:
            reason = "rolling fresh replay does not contain win, draw, and loss rows"
            accepted.append(
                {
                    "update": update,
                    "accepted": False,
                    "rollback": True,
                    "reasons": [reason],
                    "teacher_sha256": teacher_sha256,
                    "selfplay": replay_metrics,
                    "replay_path": str(replay_path),
                    "replay_header": asdict(replay_header),
                    "rolling_value_outcomes": sorted(fresh_outcomes),
                }
            )
            stopped_reason = f"update-{update:03d}-incomplete-fresh-outcome-coverage"
            break
        fresh_wdl_inputs, _ = _prepare_value(fresh_value_records, rules)
        fresh_wdl_labels = mx.array(
            [{1: 0, 0: 1, -1: 2}[int(record.outcome_value)] for record in fresh_value_records],
            dtype=mx.int32,
        )
        snapshot = replace(
            snapshot,
            replay_samples=len(fresh_value_records),
            replay_capacity=sum(len(generation) for generation in rolling_value_records),
        )
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.EVALUATION,
            mode_detail=f"DEVRIYE latest teacher targets · update {update}/{config.updates}",
            active_games=config.train_targets_per_update + config.validation_targets_per_update,
        )
        store.write_atomic(snapshot)
        target_rows, target_metrics = _generate_targets(
            previous_network,
            selected,
            config=config,
            update=update,
        )
        target_path = config.output_dir / "targets" / f"update-{update:03d}.json"
        _atomic_json(
            target_path,
            {
                "update": update,
                "teacher_sha256": teacher_sha256,
                "algorithm": "full-gumbel-mctx-style-v1",
                "simulations": config.simulations,
                "metrics": target_metrics,
                "rows": target_rows,
            },
        )
        target_artifacts.append(str(target_path))
        prepared_train = _prepare(selected["train"], target_rows["train"], rules=rules)
        prepared_validation = _prepare(
            selected["validation"], target_rows["validation"], rules=rules
        )
        rolling.append(prepared_train)
        rolling = rolling[-config.rolling_generations :]
        policy_buffer = _combine_policy(rolling)
        before_policy_logits = previous_network(prepared_validation.inputs)[0]
        mx.eval(before_policy_logits)
        before_policy = _policy_quality(
            before_policy_logits,
            prepared_validation.targets,
            prepared_validation.legal_masks,
        )

        policy_rng = random.Random(config.seed + update * 10)
        historical_value_sampler = _MixedWDLSampler(
            train_records, seed=config.seed + update * 10 + 1
        )
        fresh_value_sampler = FixedOutcomeRatioGameBalancedSampler(
            fresh_value_records,
            seed=config.seed + update * 10 + 2,
            outcome_counts={-1: 8, 0: 16, 1: 8},
        )
        curve = []
        validation_checkpoints = []
        value_validation_checkpoints = []
        value_checkpoint_found = False
        maximum_gradient = 0.0
        snapshot = replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.TRAINING,
            mode_detail=f"DEVRIYE continuous learner · update {update}/{config.updates}",
            active_games=0,
        )
        store.write_atomic(snapshot)
        for local_step in range(1, config.steps_per_update + 1):
            policy_indices = tuple(
                policy_rng.randrange(len(policy_buffer.records)) for _ in range(config.batch_size)
            )
            half_value_batch = config.batch_size // 2
            historical_value_indices = historical_value_sampler.sample_indices(half_value_batch)
            fresh_value_indices = fresh_value_sampler.sample_indices(half_value_batch)
            policy_rows = mx.array(policy_indices, dtype=mx.int32)
            historical_value_rows = mx.array(historical_value_indices, dtype=mx.int32)
            fresh_value_rows = mx.array(fresh_value_indices, dtype=mx.int32)
            value_inputs = mx.concatenate(
                (
                    mx.take(wdl_train_inputs, historical_value_rows, axis=0),
                    mx.take(fresh_wdl_inputs, fresh_value_rows, axis=0),
                ),
                axis=0,
            )
            value_targets = mx.concatenate(
                (
                    mx.take(wdl_labels, historical_value_rows, axis=0),
                    mx.take(fresh_wdl_labels, fresh_value_rows, axis=0),
                ),
                axis=0,
            )
            metric = learner.train_step(
                mx.take(policy_buffer.inputs, policy_rows, axis=0),
                mx.take(policy_buffer.targets, policy_rows, axis=0),
                mx.take(policy_buffer.legal_masks, policy_rows, axis=0),
                value_inputs,
                value_targets,
            )
            maximum_gradient = max(maximum_gradient, float(metric["gradient_norm"]))
            if not value_checkpoint_found:
                value_validation = _wdl_quality(
                    network, wdl_validation_inputs, validation_outcomes
                )
                value_reasons = _continuous_wdl_gate(previous_wdl, value_validation)
                value_validation_checkpoints.append(
                    {
                        "local_step": local_step,
                        "learner_step": learner.step,
                        "wdl": value_validation,
                        "reasons": value_reasons,
                        "state": learner.snapshot(),
                    }
                )
                value_checkpoint_found = not value_reasons
            validation_payload = None
            if local_step % 10 == 0:
                validation_policy_logits = network(prepared_validation.inputs)[0]
                mx.eval(validation_policy_logits)
                validation_policy = _policy_quality(
                    validation_policy_logits,
                    prepared_validation.targets,
                    prepared_validation.legal_masks,
                )
                validation_wdl = _wdl_quality(network, wdl_validation_inputs, validation_outcomes)
                numeric_reasons = (
                    *_policy_gate(before_policy, validation_policy),
                    *_continuous_wdl_gate(previous_wdl, validation_wdl),
                )
                validation_payload = {
                    "policy": validation_policy,
                    "wdl": validation_wdl,
                    "reasons": numeric_reasons,
                }
                validation_checkpoints.append(
                    {
                        "local_step": local_step,
                        "learner_step": learner.step,
                        **validation_payload,
                        "state": learner.snapshot(),
                    }
                )
                elapsed = time.perf_counter() - started
                history = HistoryPoint(
                    training_step=learner.step,
                    training_elapsed_seconds=elapsed,
                    lifetime_games=snapshot.lifetime_games,
                    total_loss=float(metric["total_loss"]),
                    elo_delta=None,
                    elo_low=None,
                    elo_high=None,
                    games_per_hour=snapshot.games_per_hour,
                    positions_per_second=(learner.step * config.batch_size / max(elapsed, 1e-9)),
                    policy_loss=float(metric["policy_loss"]),
                    value_loss=float(metric["value_loss"]),
                    validation_loss=float(validation_policy["cross_entropy"])
                    + float(validation_wdl["cross_entropy"]),
                )
                snapshot = replace(
                    snapshot,
                    updated_at=datetime.now(UTC).isoformat(),
                    pilot_steps_completed=learner.step,
                    training_step=learner.step,
                    policy_loss=float(metric["policy_loss"]),
                    value_loss=float(metric["value_loss"]),
                    total_loss=float(metric["total_loss"]),
                    history=(*snapshot.history, history)[-240:],
                )
                store.write_atomic(snapshot)
            curve.append({**metric, "validation": validation_payload})

        policy_checkpoints = tuple(
            checkpoint
            for checkpoint in validation_checkpoints
            if not _policy_gate(before_policy, checkpoint["policy"])
        )
        value_checkpoints = tuple(
            checkpoint
            for checkpoint in value_validation_checkpoints
            if not checkpoint["reasons"]
        )
        policy_checkpoint = min(
            policy_checkpoints or validation_checkpoints,
            key=lambda checkpoint: int(checkpoint["local_step"]),
        )
        value_checkpoint = min(
            value_checkpoints or value_validation_checkpoints,
            key=lambda checkpoint: int(checkpoint["local_step"]),
        )
        learner.restore(_compose_headwise_state(policy_checkpoint, value_checkpoint))
        learner.reset_optimizer()
        composed_policy_logits = network(prepared_validation.inputs)[0]
        mx.eval(composed_policy_logits)
        after_policy = _policy_quality(
            composed_policy_logits,
            prepared_validation.targets,
            prepared_validation.legal_masks,
        )
        after_wdl = _wdl_quality(network, wdl_validation_inputs, validation_outcomes)
        numeric_reasons = (
            *_policy_gate(before_policy, after_policy),
            *_continuous_wdl_gate(previous_wdl, after_wdl),
        )
        if not policy_checkpoints:
            numeric_reasons = (
                "no validation checkpoint independently passed policy gates",
                *numeric_reasons,
            )
        if not value_checkpoints:
            numeric_reasons = (
                "no validation checkpoint independently passed WDL gates",
                *numeric_reasons,
            )
        after_material = _material_quality(network, *material_validation)
        continuation = _continuation_ranking(
            release,
            network,
            ranking_records,
            rules=rules,
            depth=config.ranking_depth,
        )
        tactical = _tactical(_clone(network), config=tactical_config)
        tactical_reasons = _tactical_gate(initial_tactical, tactical)
        if int(tactical["budgets"][0]["solved"]) < 5:  # type: ignore[index]
            tactical_reasons = (*tactical_reasons, "Full Gumbel tactical is below 5/8")
        arena = _paired_arena(
            previous_network,
            _clone(network),
            pairs=config.arena_pairs_per_update,
            simulations=config.arena_simulations_per_update,
            seed=config.seed + 1000 + update,
            config=config,
        )
        reasons = [
            *numeric_reasons,
            *_continuation_floor(continuation),
            *tactical_reasons,
        ]
        if after_material != material_baseline:
            reasons.append("auxiliary material predictions changed")
        if _parameter_hash(network, excluded_prefixes=_TRAINABLE_PREFIXES) != nontrainable_hash:
            reasons.append("continuous learner changed a frozen parameter")
        if maximum_gradient > 5.0:
            reasons.append("gradient norm exceeded 5.0")
        if float(arena["score_rate"]) < 0.375:
            reasons.append("per-update search score is below catastrophic floor 0.375")
        accepted_update = not reasons
        checkpoint_path = (
            config.output_dir / "checkpoints" / f"update-{update:03d}" / "model.safetensors"
        )
        checkpoint_sha256 = None
        if accepted_update:
            checkpoint_sha256 = _save_network(network, checkpoint_path)
        else:
            learner.restore(before_state)
            rolling.pop()
            stopped_reason = f"update-{update:03d}-rollback"
        accepted.append(
            {
                "update": update,
                "accepted": accepted_update,
                "rollback": not accepted_update,
                "reasons": reasons,
                "teacher_sha256": teacher_sha256,
                "target_path": str(target_path),
                "target_metrics": target_metrics,
                "selfplay": replay_metrics,
                "replay_path": str(replay_path),
                "replay_header": asdict(replay_header),
                "rolling_value_generations": len(rolling_value_records),
                "rolling_value_rows": len(fresh_value_records),
                "value_batch_mix": {"historical": 32, "fresh": 32},
                "fresh_value_sampling": "fixed-8-loss-16-draw-8-win-then-game-balanced",
                "rolling_policy_rows": len(policy_buffer.records),
                "steps": config.steps_per_update,
                "selected_local_step": max(
                    int(policy_checkpoint["local_step"]),
                    int(value_checkpoint["local_step"]),
                ),
                "selected_policy_local_step": policy_checkpoint["local_step"],
                "selected_wdl_local_step": value_checkpoint["local_step"],
                "optimizer_reset_after_composition": True,
                "learner_step": learner.step,
                "maximum_gradient_norm": maximum_gradient,
                "policy_before": before_policy,
                "policy_after": after_policy,
                "wdl_before": previous_wdl,
                "wdl_after": after_wdl,
                "material": after_material,
                "continuation": continuation,
                "tactical": tactical,
                "arena": arena,
                "checkpoint_path": str(checkpoint_path) if accepted_update else None,
                "checkpoint_sha256": checkpoint_sha256,
                "validation_checkpoints": [
                    {key: value for key, value in checkpoint.items() if key != "state"}
                    for checkpoint in validation_checkpoints
                ],
                "value_validation_checkpoints": [
                    {key: value for key, value in checkpoint.items() if key != "state"}
                    for checkpoint in value_validation_checkpoints
                ],
                "curve": curve,
            }
        )
        if not accepted_update:
            break

    final_arena = None
    chain_reasons = []
    if len(accepted) != config.updates or not all(row["accepted"] for row in accepted):
        chain_reasons.append("not all preregistered continuous updates were accepted")
    else:
        final_arena = _paired_arena(
            initial_network,
            _clone(network),
            pairs=config.final_arena_pairs,
            simulations=config.final_arena_simulations,
            seed=config.seed + 2000,
            config=config,
        )
        final_wdl = accepted[-1]["wdl_after"]
        final_continuation = accepted[-1]["continuation"]
        if float(final_arena["score_rate"]) < 0.50:
            chain_reasons.append("final search score against MIHVER start is below 0.50")
        if float(final_wdl["cross_entropy"]) > float(initial_wdl["cross_entropy"]):
            chain_reasons.append("final WDL micro CE regressed versus MIHVER start")
        if float(final_wdl["macro_cross_entropy"]) > float(initial_wdl["macro_cross_entropy"]):
            chain_reasons.append("final WDL macro CE regressed versus MIHVER start")
        if float(final_wdl["expected_score_pearson"]) < float(
            initial_wdl["expected_score_pearson"]
        ):
            chain_reasons.append("final WDL Pearson regressed versus MIHVER start")
        if float(final_continuation["candidate_mean_spearman"]) < float(
            initial_continuation["candidate_mean_spearman"]
        ):
            chain_reasons.append("final continuation Spearman regressed versus MIHVER start")
        if float(final_continuation["candidate_verified_top_agreement"]) < float(
            initial_continuation["candidate_verified_top_agreement"]
        ):
            chain_reasons.append("final continuation top agreement regressed versus MIHVER start")
    passed = not chain_reasons
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
                        "runs_root",
                        "telemetry_path",
                    )
                },
            },
            "provenance": provenance,
            "split": split,
            "initial": {
                "source_path": str(initial_path),
                "checkpoint_path": str(initial_checkpoint),
                "sha256": initial_sha256,
                "wdl": initial_wdl,
                "material": material_baseline,
                "continuation": initial_continuation,
                "tactical": initial_tactical,
            },
            "updates": accepted,
            "target_artifacts": target_artifacts,
            "stopped_reason": stopped_reason,
            "final_arena": final_arena,
            "passed": passed,
            "reasons": chain_reasons,
            "continuous_generation_authorized": passed,
            "promotion_architecture_authorized": passed,
            "promotion_authorized": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    detail = (
        "DEVRIYE continuous pilot passed · generation integration authorized"
        if passed
        else "DEVRIYE continuous pilot failed · latest rolled back"
    )
    final_checkpoint = accepted[-1].get("checkpoint_path") if passed else None
    store.write_atomic(
        replace(
            snapshot,
            updated_at=datetime.now(UTC).isoformat(),
            mode=RunMode.IDLE,
            mode_detail=detail,
            pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
            pilot_stop_reason="continuous_chain_gate",
            pilot_stop_detail=detail,
            pilot_reasons=tuple(chain_reasons),
            candidate_checkpoint=(
                Path(str(final_checkpoint)).parent.name if final_checkpoint else "None"
            ),
            checkpoint_status=(CheckpointStatus.VERIFIED if passed else CheckpointStatus.NONE),
            checkpoint_path=str(final_checkpoint or ""),
            checkpoint_verified=passed,
            promotion_ready=False,
        )
    )
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--value-result", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument("--seed", type=int, default=2026083101)
    arguments = parser.parse_args(argv)
    result = run_continuous_policy_iteration(
        ContinuousPolicyIterationConfig(
            output_dir=arguments.output_dir,
            value_result=arguments.value_result,
            model_path=arguments.model,
            runs_root=arguments.runs_root,
            telemetry_path=arguments.telemetry,
            seed=arguments.seed,
        )
    )
    print(result)
    return 0 if json.loads(result.read_text(encoding="utf-8"))["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
