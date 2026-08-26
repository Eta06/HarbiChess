"""Run a low-budget OCAK self-play and learner sanity cycle with live telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import TerminalResult
from harbichess.dashboard.state import (
    CheckpointStatus,
    DiversitySnapshot,
    HistoryPoint,
    LiveGame,
    OpeningDiversity,
    PilotStatus,
    RunMode,
    SnapshotStore,
    TerminationSnapshot,
    empty_snapshot,
)
from harbichess.replay.diversity import DiversityMetrics, measure_diversity
from harbichess.replay.merge import merge_continuation_replay
from harbichess.replay.schema import ReplayRecord, records_from_game
from harbichess.replay.shard import ShardMetadata, read_shard, write_shard_atomic
from harbichess.replay.split import ReplaySplit, partition_games
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.mcts import MCTS, SearchConfig
from harbichess.search.root_halving import RootHalvingConfig
from harbichess.selfplay.game import SelfPlayConfig, SelfPlayGame, play_parallel_games
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from harbichess.training.learner import LearnerConfig, MLXLearner, TrainingMetrics
from harbichess.training.pilot import PilotConfig, run_sanity_pilot
from harbichess.training.resume import ResumeState


@dataclass(frozen=True, slots=True)
class OcakRunConfig:
    run_id: str
    artifact_root: Path = Path("artifacts/runs")
    telemetry_path: Path = Path("artifacts/dashboard/state.json")
    run_seed: int = 20260824
    games: int = 12
    workers: int = 12
    simulations: int = 8
    max_plies: int = 64
    exploration_plies: int = 30
    validation_fraction: float = 0.25
    training_steps: int = 40
    batch_size: int = 16
    learning_rate: float = 0.002
    minimum_train_improvement: float = 0.02
    maximum_validation_ratio: float = 1.25
    minimum_decisive_games: int = 1
    maximum_max_ply_draw_ratio: float = 0.9
    maximum_repetition_draw_ratio: float = 0.5
    telemetry_interval_steps: int = 2
    validation_interval_steps: int = 10
    early_stopping_patience: int = 12
    minimum_validation_delta: float = 1e-3
    trunk_channels: int = 16
    residual_blocks: int = 2
    policy_channels: int = 4
    value_channels: int = 2
    value_hidden: int = 32
    continuation_shards: tuple[Path, ...] = ()
    continuation_batch_fraction: float = 0.25
    initial_model: Path | None = None
    inference_wait_seconds: float = 0.00025
    continuation_recency_decay: float = 0.60
    value_policy_temperature: float | None = None
    value_policy_prior_visits: float = 8.0
    maximum_value_logit_adjustment: float = 1.25
    root_halving_enabled: bool = False
    root_halving_top_actions: int = 4
    root_halving_finalists: int = 2
    root_halving_first_round_simulations: int = 3
    root_halving_final_round_simulations: int = 7
    root_halving_minimum_margin: float = 0.05
    root_halving_transfer_fraction: float = 0.35
    replay_split_namespace: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be one safe path segment")
        positive = (
            self.games,
            self.workers,
            self.simulations,
            self.max_plies,
            self.training_steps,
            self.batch_size,
            self.telemetry_interval_steps,
            self.validation_interval_steps,
            self.early_stopping_patience,
            self.trunk_channels,
            self.residual_blocks,
            self.policy_channels,
            self.value_channels,
            self.value_hidden,
        )
        if any(value <= 0 for value in positive) or self.exploration_plies < 0:
            raise ValueError("OCAK run counts and network dimensions must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if self.run_seed < 0 or self.learning_rate <= 0:
            raise ValueError("run seed must be non-negative and learning rate positive")
        if self.minimum_decisive_games < 0:
            raise ValueError("minimum_decisive_games must be non-negative")
        if not math.isfinite(self.minimum_validation_delta) or self.minimum_validation_delta < 0:
            raise ValueError("minimum_validation_delta must be finite and non-negative")
        if not 0.0 <= self.maximum_max_ply_draw_ratio <= 1.0:
            raise ValueError("maximum_max_ply_draw_ratio must be in [0, 1]")
        if not 0.0 <= self.maximum_repetition_draw_ratio <= 1.0:
            raise ValueError("maximum_repetition_draw_ratio must be in [0, 1]")
        if not 0.0 <= self.continuation_batch_fraction <= 1.0:
            raise ValueError("continuation_batch_fraction must be in [0, 1]")
        if self.inference_wait_seconds < 0:
            raise ValueError("inference_wait_seconds must be non-negative")
        if not 0.0 < self.continuation_recency_decay <= 1.0:
            raise ValueError("continuation_recency_decay must be in (0, 1]")
        if self.value_policy_temperature is not None and self.value_policy_temperature <= 0:
            raise ValueError("value_policy_temperature must be positive when enabled")
        if self.value_policy_prior_visits < 0 or self.maximum_value_logit_adjustment < 0:
            raise ValueError("value-policy shrinkage and logit bounds must be non-negative")
        if self.root_halving_enabled and self.value_policy_temperature is not None:
            raise ValueError("root halving and value-policy reweighting are mutually exclusive")
        if self.replay_split_namespace is not None and (
            not self.replay_split_namespace
            or Path(self.replay_split_namespace).name != self.replay_split_namespace
        ):
            raise ValueError("replay_split_namespace must be one safe path segment")
        if self.root_halving_enabled:
            root_halving = RootHalvingConfig(
                top_actions=self.root_halving_top_actions,
                finalists=self.root_halving_finalists,
                first_round_simulations=self.root_halving_first_round_simulations,
                final_round_simulations=self.root_halving_final_round_simulations,
                minimum_consensus_margin=self.root_halving_minimum_margin,
                transfer_fraction=self.root_halving_transfer_fraction,
            )
            if self.simulations <= root_halving.forced_evaluations:
                raise ValueError("total simulations must exceed forced root evaluations")


@dataclass(frozen=True, slots=True)
class OcakRunResult:
    run_id: str
    source_commit: str
    passed: bool
    reasons: tuple[str, ...]
    games: int
    replay_samples: int
    validation_samples: int
    training_steps: int
    self_play_seconds: float
    training_seconds: float
    total_seconds: float
    checkpoint_path: str
    result_path: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _diversity_snapshot(metrics: DiversityMetrics) -> DiversitySnapshot:
    return DiversitySnapshot(
        games=metrics.games,
        positions=metrics.positions,
        unique_game_ratio=metrics.unique_game_ratio,
        duplicate_game_ratio=metrics.duplicate_game_ratio,
        unique_position_ratio=metrics.unique_position_ratio,
        selected_actions=metrics.selected_actions,
        action_space_coverage=metrics.action_space_coverage,
        mean_policy_entropy=metrics.mean_policy_entropy,
        effective_policy_branches=metrics.effective_policy_branches,
        mean_game_plies=metrics.mean_game_plies,
        white_wins=metrics.white_wins,
        draws=metrics.draws,
        black_wins=metrics.black_wins,
        decisive_games=metrics.decisive_games,
        decisive_game_ratio=metrics.decisive_game_ratio,
        max_ply_draws=metrics.max_ply_draws,
        max_ply_draw_ratio=metrics.max_ply_draw_ratio,
        repetition_redirects=metrics.repetition_redirects,
        repetition_redirect_ratio=metrics.repetition_redirect_ratio,
        terminations=tuple(
            TerminationSnapshot(item.termination, item.count, item.ratio)
            for item in metrics.terminations
        ),
        openings=tuple(
            OpeningDiversity(
                ply=opening.ply,
                eligible_games=opening.eligible_games,
                unique_prefixes=opening.unique_prefixes,
                entropy=opening.entropy,
                effective_prefixes=opening.effective_prefixes,
            )
            for opening in metrics.openings
        ),
    )


def _latest_game(game: SelfPlayGame) -> LiveGame:
    if not game.samples:
        return LiveGame(game_id=f"game-{game.game_index}")
    sample = game.samples[-1]
    policy = sorted(sample.visit_policy, key=lambda item: (-item[1], item[0].uci))[:3]
    value = sample.root_value
    wdl = (max(0.0, value), max(0.0, 1.0 - abs(value)), max(0.0, -value))
    return LiveGame(
        game_id=f"game-{game.game_index}",
        fen=game.final_state.root_fen
        if not game.final_state.moves
        else PythonChessRules().board(game.final_state).fen(),
        last_move=game.final_state.moves[-1].uci if game.final_state.moves else "",
        ply=game.final_state.ply,
        top_moves=tuple((move.uci, probability) for move, probability in policy),
        wdl=wdl,
    )


def _records(
    games: tuple[SelfPlayGame, ...],
    *,
    run_id: str,
    rules: PythonChessRules,
) -> tuple[ReplayRecord, ...]:
    return tuple(
        record for game in games for record in records_from_game(game, run_id=run_id, rules=rules)
    )


def run_ocak_sanity(
    config: OcakRunConfig,
    *,
    source_commit: str | None = None,
) -> OcakRunResult:
    commit = source_commit or _source_commit()
    if len(commit) != 40:
        raise ValueError("source_commit must be a full Git SHA")
    run_root = config.artifact_root / config.run_id
    if run_root.exists():
        raise FileExistsError(f"run already exists: {run_root}")
    store = SnapshotStore(config.telemetry_path)
    started = time.perf_counter()
    training_started: float | None = None
    lock = threading.Lock()
    snapshot = replace(
        empty_snapshot(),
        updated_at=_now(),
        mode=RunMode.SELF_PLAY,
        mode_detail=f"OCAK sanity self-play · 0/{config.games} games",
        run_id=config.run_id,
        source_commit=commit,
        active_checkpoint=(
            "provided-champion" if config.initial_model is not None else "random-initial"
        ),
        pilot_status=PilotStatus.SELF_PLAY,
        pilot_steps_planned=config.training_steps,
        pilot_validation_interval_steps=config.validation_interval_steps,
        pilot_early_stopping_patience=config.early_stopping_patience,
        active_games=config.games,
        replay_capacity=config.games * config.max_plies,
        learning_rate=config.learning_rate,
    )

    def publish(**changes: object) -> None:
        nonlocal snapshot
        with lock:
            elapsed = time.perf_counter() - started
            training_elapsed = (
                time.perf_counter() - training_started
                if training_started is not None
                else snapshot.training_elapsed_seconds
            )
            snapshot = replace(
                snapshot,
                updated_at=_now(),
                session_elapsed_seconds=elapsed,
                training_elapsed_seconds=training_elapsed,
                **changes,
            )
            store.write_atomic(snapshot)

    store.write_atomic(snapshot)
    network_config = NetworkConfig(
        trunk_channels=config.trunk_channels,
        residual_blocks=config.residual_blocks,
        policy_channels=config.policy_channels,
        value_channels=config.value_channels,
        value_hidden=config.value_hidden,
    )
    mx.random.seed(config.run_seed)
    network = HarbiChessNetwork(network_config)
    if config.initial_model is not None:
        if not config.initial_model.is_file():
            raise FileNotFoundError(f"initial model does not exist: {config.initial_model}")
        network.load_weights(str(config.initial_model))
    initial_checkpoint = (
        "provided-champion" if config.initial_model is not None else "random-initial"
    )
    baseline_dir = run_root / "baseline"
    baseline_dir.mkdir(parents=True)
    baseline_path = baseline_dir / "model.safetensors"
    temporary_baseline = baseline_dir / ".model.tmp.safetensors"
    network.save_weights(str(temporary_baseline))
    with temporary_baseline.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary_baseline, baseline_path)
    baseline_sha256 = _sha256(baseline_path)
    rules = PythonChessRules()
    completed_games: list[SelfPlayGame] = []
    completed_positions = 0
    self_play_started = time.perf_counter()
    batcher: SharedBatchEvaluator | None = None
    inference_statistics = None
    try:
        backend = MLXPolicyValueBackend(network)
        batcher = SharedBatchEvaluator(
            backend,
            max_batch_size=max(1, min(128, config.workers * 2)),
            max_wait_seconds=config.inference_wait_seconds,
        )
        root_halving_config = (
            RootHalvingConfig(
                top_actions=config.root_halving_top_actions,
                finalists=config.root_halving_finalists,
                first_round_simulations=config.root_halving_first_round_simulations,
                final_round_simulations=config.root_halving_final_round_simulations,
                minimum_consensus_margin=config.root_halving_minimum_margin,
                transfer_fraction=config.root_halving_transfer_fraction,
            )
            if config.root_halving_enabled
            else None
        )
        initial_simulations = (
            config.simulations - root_halving_config.forced_evaluations
            if root_halving_config is not None
            else config.simulations
        )
        search = MCTS(
            NeuralPositionEvaluator(batcher, rules=rules),
            rules=rules,
            config=SearchConfig(simulations=initial_simulations),
        )

        def game_complete(game: SelfPlayGame) -> None:
            nonlocal completed_positions
            with lock:
                completed_games.append(game)
                completed_positions += len(game.samples)
                count = len(completed_games)
                positions = completed_positions
                outcomes = Counter(item.outcome.result for item in completed_games)
                terminations = Counter(item.outcome.termination for item in completed_games)
                white_wins = outcomes[TerminalResult.WHITE_WIN]
                black_wins = outcomes[TerminalResult.BLACK_WIN]
                draws = outcomes[TerminalResult.DRAW]
                max_ply_draws = terminations["max_plies"]
                live_diversity = replace(
                    snapshot.diversity,
                    games=count,
                    positions=positions,
                    mean_game_plies=(sum(item.final_state.ply for item in completed_games) / count),
                    white_wins=white_wins,
                    draws=draws,
                    black_wins=black_wins,
                    decisive_games=white_wins + black_wins,
                    decisive_game_ratio=(white_wins + black_wins) / count,
                    max_ply_draws=max_ply_draws,
                    max_ply_draw_ratio=max_ply_draws / count,
                    terminations=tuple(
                        TerminationSnapshot(termination, amount, amount / count)
                        for termination, amount in sorted(terminations.items())
                    ),
                )
            elapsed = max(time.perf_counter() - self_play_started, 1e-9)
            statistics = batcher.statistics if batcher is not None else None
            publish(
                mode_detail=f"OCAK sanity self-play · {count}/{config.games} games",
                active_games=config.games - count,
                completed_games=count,
                run_games=count,
                generation_games=count,
                lifetime_games=count,
                replay_samples=positions,
                games_per_hour=count / elapsed * 3600.0,
                positions_per_second=positions / elapsed,
                neural_evals_per_second=(statistics.positions / elapsed if statistics else 0.0),
                mcts_nodes_per_second=positions * config.simulations / elapsed,
                inference_batch_size=(round(statistics.average_batch_size) if statistics else 0),
                live_game=_latest_game(game),
                diversity=live_diversity,
            )

        games = play_parallel_games(
            search,
            rules,
            run_seed=config.run_seed,
            first_game_index=0,
            game_count=config.games,
            max_workers=min(config.workers, config.games),
            config=SelfPlayConfig(
                exploration_plies=config.exploration_plies,
                max_plies=config.max_plies,
                value_policy_temperature=config.value_policy_temperature,
                value_policy_prior_visits=config.value_policy_prior_visits,
                maximum_value_logit_adjustment=(
                    config.maximum_value_logit_adjustment
                ),
                root_halving_config=root_halving_config,
            ),
            on_game_complete=game_complete,
        )
        self_play_seconds = time.perf_counter() - self_play_started
        inference_statistics = batcher.statistics
        batcher.close()
        batcher = None

        diversity_metrics = measure_diversity(games)
        partitions = partition_games(
            games,
            run_id=config.replay_split_namespace or config.run_id,
            validation_fraction=config.validation_fraction,
        )
        train_games = partitions[ReplaySplit.TRAIN]
        validation_games = partitions[ReplaySplit.VALIDATION]
        if not train_games or not validation_games:
            raise RuntimeError(
                "deterministic game split produced an empty train or validation partition"
            )
        generated_train_records = _records(
            train_games,
            run_id=config.run_id,
            rules=rules,
        )
        validation_records = _records(validation_games, run_id=config.run_id, rules=rules)
        continuation_shards = tuple(
            read_shard(path, rules=rules) for path in config.continuation_shards
        )
        if any(shard.header.split is not ReplaySplit.TRAIN for shard in continuation_shards):
            raise ValueError("continuation replay shards must use the train split")
        continuation_merge = merge_continuation_replay(
            tuple(zip(config.continuation_shards, continuation_shards, strict=True)),
            recency_decay=config.continuation_recency_decay,
        )
        continuation_records = continuation_merge.records
        train_records = (*generated_train_records, *continuation_records)
        if {record.game_id for record in train_records} & {
            record.game_id for record in validation_records
        }:
            raise ValueError("continuation replay leaks validation game IDs")
        replay_dir = run_root / "replay"
        created_at = _now()
        train_header = write_shard_atomic(
            replay_dir / "train-00000.jsonl.gz",
            generated_train_records,
            ShardMetadata(
                run_id=config.run_id,
                generation=0,
                source_checkpoint=initial_checkpoint,
                source_commit=commit,
                created_at=created_at,
                split=ReplaySplit.TRAIN,
            ),
        )
        validation_header = write_shard_atomic(
            replay_dir / "validation-00000.jsonl.gz",
            validation_records,
            ShardMetadata(
                run_id=config.run_id,
                generation=0,
                source_checkpoint=initial_checkpoint,
                source_commit=commit,
                created_at=created_at,
                split=ReplaySplit.VALIDATION,
            ),
        )
        publish(
            mode=RunMode.TRAINING,
            mode_detail="Replay verified · preparing learner pilot",
            pilot_status=PilotStatus.REPLAY,
            active_games=0,
            replay_samples=len(train_records) + len(validation_records),
            validation_samples=len(validation_records),
            replay_shards=2 + len(continuation_shards),
            continuation_replay_samples=len(continuation_records),
            diversity=_diversity_snapshot(diversity_metrics),
        )

        network.set_dtype(mx.float32)
        network.train()
        learner_config = LearnerConfig(
            learning_rate=config.learning_rate,
            weight_decay=0.0,
            max_gradient_norm=5.0,
        )
        learner = MLXLearner(network, config=learner_config)
        train_evaluation = build_training_batch(train_records)
        validation_evaluation = build_training_batch(validation_records)
        initial_train = learner.evaluate_loss(train_evaluation)[0]
        initial_validation = learner.evaluate_loss(validation_evaluation)[0]
        training_started = time.perf_counter()
        latest_validation_loss = initial_validation
        publish(
            mode=RunMode.TRAINING,
            mode_detail=f"OCAK learner pilot · 0/{config.training_steps} steps",
            pilot_status=PilotStatus.TRAINING,
            pilot_initial_train_loss=initial_train,
            pilot_initial_validation_loss=initial_validation,
        )

        def training_step(
            metric: TrainingMetrics,
            measured_validation_loss: float | None,
        ) -> None:
            nonlocal latest_validation_loss
            if measured_validation_loss is not None:
                latest_validation_loss = measured_validation_loss
            if (
                metric.step % config.telemetry_interval_steps != 0
                and metric.step != config.training_steps
            ):
                return
            elapsed = max(time.perf_counter() - training_started, 1e-9)
            point = HistoryPoint(
                training_step=metric.step,
                training_elapsed_seconds=elapsed,
                lifetime_games=config.games,
                total_loss=metric.total_loss,
                elo_delta=None,
                elo_low=None,
                elo_high=None,
                games_per_hour=snapshot.games_per_hour,
                positions_per_second=metric.step * config.batch_size / elapsed,
                policy_loss=metric.policy_loss,
                value_loss=metric.value_loss,
                validation_loss=latest_validation_loss,
            )
            publish(
                mode_detail=(f"OCAK learner pilot · {metric.step}/{config.training_steps} steps"),
                training_step=metric.step,
                pilot_steps_completed=metric.step,
                policy_loss=metric.policy_loss,
                value_loss=metric.value_loss,
                total_loss=metric.total_loss,
                positions_per_second=point.positions_per_second,
                pilot_max_gradient_norm=max(
                    snapshot.pilot_max_gradient_norm or 0.0,
                    metric.gradient_norm,
                ),
                history=(*snapshot.history, point)[-240:],
            )

        report = run_sanity_pilot(
            learner,
            train_records,
            validation_records,
            config=PilotConfig(
                steps=config.training_steps,
                batch_size=config.batch_size,
                minimum_train_improvement=config.minimum_train_improvement,
                maximum_validation_ratio=config.maximum_validation_ratio,
                validation_interval_steps=config.validation_interval_steps,
                early_stopping_patience=config.early_stopping_patience,
                minimum_validation_delta=config.minimum_validation_delta,
                seed=config.run_seed,
                continuation_fraction=(
                    config.continuation_batch_fraction if continuation_records else None
                ),
                continuation_game_weights=(
                    continuation_merge.game_weights if continuation_records else None
                ),
            ),
            on_step=training_step,
            train_evaluation=train_evaluation,
            validation_evaluation=validation_evaluation,
        )
        outcome_reasons = []
        if diversity_metrics.decisive_games < config.minimum_decisive_games:
            outcome_reasons.append("self-play did not produce the required decisive terminal games")
        if diversity_metrics.max_ply_draw_ratio > config.maximum_max_ply_draw_ratio:
            outcome_reasons.append("too many self-play games ended at the max-ply limit")
        repetition_draws = sum(
            termination.count
            for termination in diversity_metrics.terminations
            if termination.termination == "threefold_repetition"
        )
        if repetition_draws / diversity_metrics.games > config.maximum_repetition_draw_ratio:
            outcome_reasons.append("too many self-play games ended by threefold repetition")
        reasons = (*report.reasons, *outcome_reasons)
        passed = report.passed and not outcome_reasons
        training_seconds = time.perf_counter() - training_started
        stop_detail = (
            f"No validation improvement for {report.stale_validation_evaluations} "
            "evaluations "
            f"({report.stale_validation_evaluations * config.validation_interval_steps} "
            f"steps); restored step {report.best_validation_step}"
            if report.stopped_early
            else f"Reached configured maximum of {config.training_steps} steps"
        )
        publish(
            mode=RunMode.CHECKPOINTING,
            mode_detail="Pilot complete · writing atomic candidate checkpoint",
            pilot_status=PilotStatus.PASSED if passed else PilotStatus.FAILED,
            training_step=report.steps,
            pilot_steps_completed=report.steps,
            pilot_steps_attempted=report.attempted_steps,
            pilot_best_validation_step=report.best_validation_step,
            pilot_best_validation_loss=report.best_validation_loss,
            pilot_stopped_early=report.stopped_early,
            pilot_stop_reason=report.stop_reason,
            pilot_stop_detail=stop_detail,
            pilot_last_validation_step=report.last_validation_step,
            pilot_last_validation_loss=report.last_validation_loss,
            pilot_last_improvement_step=report.last_improvement_step,
            pilot_stale_validation_evaluations=report.stale_validation_evaluations,
            pilot_validation_evaluations=report.validation_evaluations,
            pilot_final_train_loss=report.final_train_loss,
            pilot_final_validation_loss=report.final_validation_loss,
            pilot_max_gradient_norm=report.maximum_gradient_norm,
            pilot_reasons=reasons,
            checkpoint_status=CheckpointStatus.WRITING,
            validation_checkpoint_count=len(report.validation_candidates),
        )

        validation_checkpoints = []

        def save_candidate(step: int, validation_loss: float, rng_state: object):
            checkpoint_id = f"candidate-step-{step:06d}"
            checkpoint_path = run_root / "checkpoints" / checkpoint_id
            checkpoint_sampler = GameBalancedSampler(train_records, seed=config.run_seed)
            checkpoint_sampler.set_rng_state(rng_state)
            resume = ResumeState(
                schema_version=1,
                run_id=config.run_id,
                checkpoint_id=checkpoint_id,
                source_commit=commit,
                created_at=_now(),
                training_step=step,
                lifetime_games=config.games,
                generation_games=config.games,
                training_elapsed_seconds=training_seconds,
                replay_samples=len(train_records) + len(validation_records),
                replay_cursor=len(train_records) - 1,
                model_file="model.safetensors",
                optimizer_file="optimizer.safetensors",
                rng_file="sampler-rng.json",
            )
            saved = save_training_checkpoint(
                checkpoint_path,
                state=resume,
                learner=learner,
                sampler=checkpoint_sampler,
            )
            verification_learner = MLXLearner(
                HarbiChessNetwork(network_config),
                config=learner_config,
            )
            verification_sampler = GameBalancedSampler(train_records, seed=0)
            loaded = load_training_checkpoint(
                checkpoint_path,
                learner=verification_learner,
                sampler=verification_sampler,
            )
            if loaded != saved:
                raise RuntimeError("checkpoint verification did not reproduce its resume manifest")
            entry = {
                "step": step,
                "validation_loss": validation_loss,
                "path": str(checkpoint_path),
                "verified": True,
                "manifest": asdict(saved),
            }
            validation_checkpoints.append(entry)
            return checkpoint_path, saved

        for candidate in report.validation_candidates:
            learner.restore(candidate.learner_snapshot)
            save_candidate(
                candidate.step,
                candidate.validation_loss,
                candidate.sampler_rng_state,
            )
        if not validation_checkpoints:
            checkpoint_path, saved_resume = save_candidate(
                learner.step,
                report.final_validation_loss,
                report.sampler_rng_state,
            )
        else:
            selected = next(
                item
                for item in validation_checkpoints
                if item["step"] == report.best_validation_step
            )
            checkpoint_path = Path(selected["path"])
            saved_resume = ResumeState(**selected["manifest"])
        learner.restore(
            next(
                candidate.learner_snapshot
                for candidate in report.validation_candidates
                if candidate.step == report.best_validation_step
            )
            if report.validation_candidates
            else learner.snapshot()
        )

        result_path = run_root / "result.json"
        total_seconds = time.perf_counter() - started
        result_payload = {
            "run_id": config.run_id,
            "source_commit": commit,
            "created_at": _now(),
            "passed": passed,
            "reasons": reasons,
            "config": {
                **asdict(config),
                "artifact_root": str(config.artifact_root),
                "telemetry_path": str(config.telemetry_path),
                "continuation_shards": [str(path) for path in config.continuation_shards],
                "initial_model": (
                    str(config.initial_model) if config.initial_model is not None else None
                ),
            },
            "system": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "mlx_device": mx.device_info(),
            },
            "timing": {
                "self_play_seconds": self_play_seconds,
                "training_seconds": training_seconds,
                "total_seconds": total_seconds,
            },
            "inference": {
                **asdict(inference_statistics),
                "average_batch_size": inference_statistics.average_batch_size,
                "average_queue_wait_ms": inference_statistics.average_queue_wait_ms,
            },
            "loss": {
                "initial_train": report.initial_train_loss,
                "final_train": report.final_train_loss,
                "initial_validation": report.initial_validation_loss,
                "final_validation": report.final_validation_loss,
                "maximum_gradient_norm": report.maximum_gradient_norm,
                "attempted_steps": report.attempted_steps,
                "best_validation_step": report.best_validation_step,
                "best_validation_loss": report.best_validation_loss,
                "stopped_early": report.stopped_early,
                "stop_reason": report.stop_reason,
                "stop_detail": stop_detail,
                "last_validation_step": report.last_validation_step,
                "last_validation_loss": report.last_validation_loss,
                "last_improvement_step": report.last_improvement_step,
                "stale_validation_evaluations": report.stale_validation_evaluations,
                "validation_evaluations": report.validation_evaluations,
            },
            "diversity": asdict(diversity_metrics),
            "baseline": {
                "checkpoint_id": initial_checkpoint,
                "path": str(baseline_path),
                "model_sha256": baseline_sha256,
            },
            "replay": {
                "train": asdict(train_header),
                "validation": asdict(validation_header),
                "continuation": [asdict(source) for source in continuation_merge.sources],
                "continuation_samples": len(continuation_records),
                "continuation_duplicates_removed": (continuation_merge.duplicates_removed),
                "continuation_batch_fraction": (
                    config.continuation_batch_fraction if continuation_records else 0.0
                ),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "verified": True,
                "manifest": asdict(saved_resume),
            },
            "validation_checkpoints": validation_checkpoints,
        }
        _atomic_json(result_path, result_payload)
        publish(
            mode=RunMode.IDLE,
            mode_detail=(
                "OCAK sanity pilot passed · candidate awaits DEVIR evaluation"
                if passed
                else "OCAK sanity pilot failed · champion remains unchanged"
            ),
            active_checkpoint=initial_checkpoint,
            candidate_checkpoint=saved_resume.checkpoint_id,
            promoted_checkpoint="None",
            checkpoint_status=CheckpointStatus.VERIFIED,
            checkpoint_path=str(checkpoint_path),
            checkpoint_verified=True,
            validation_checkpoint_count=len(validation_checkpoints),
        )
        return OcakRunResult(
            run_id=config.run_id,
            source_commit=commit,
            passed=passed,
            reasons=reasons,
            games=config.games,
            replay_samples=len(train_records) + len(validation_records),
            validation_samples=len(validation_records),
            training_steps=report.steps,
            self_play_seconds=self_play_seconds,
            training_seconds=training_seconds,
            total_seconds=total_seconds,
            checkpoint_path=str(checkpoint_path),
            result_path=str(result_path),
        )
    except BaseException as error:
        if batcher is not None:
            batcher.close()
        publish(
            mode=RunMode.IDLE,
            mode_detail=f"OCAK sanity run failed · {type(error).__name__}: {error}",
            pilot_status=PilotStatus.FAILED,
            pilot_reasons=(str(error),),
            active_games=0,
            failed_games=1,
            checkpoint_status=(
                CheckpointStatus.FAILED
                if snapshot.checkpoint_status is CheckpointStatus.WRITING
                else snapshot.checkpoint_status
            ),
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--telemetry", type=Path, default=Path("artifacts/dashboard/state.json"))
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=64)
    parser.add_argument("--training-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--minimum-decisive-games", type=int, default=1)
    parser.add_argument("--maximum-max-ply-draw-ratio", type=float, default=0.9)
    parser.add_argument("--maximum-repetition-draw-ratio", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--validation-interval-steps", type=int, default=10)
    parser.add_argument("--minimum-validation-delta", type=float, default=1e-3)
    parser.add_argument("--continuation-shard", action="append", type=Path, default=[])
    parser.add_argument("--continuation-batch-fraction", type=float, default=0.25)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--inference-wait-ms", type=float, default=0.25)
    parser.add_argument("--continuation-recency-decay", type=float, default=0.60)
    parser.add_argument("--value-policy-temperature", type=float)
    parser.add_argument("--value-policy-prior-visits", type=float, default=8.0)
    parser.add_argument("--maximum-value-logit-adjustment", type=float, default=1.25)
    parser.add_argument("--root-halving", action="store_true")
    parser.add_argument("--root-halving-top-actions", type=int, default=4)
    parser.add_argument("--root-halving-finalists", type=int, default=2)
    parser.add_argument("--root-halving-first-round-simulations", type=int, default=3)
    parser.add_argument("--root-halving-final-round-simulations", type=int, default=7)
    parser.add_argument("--root-halving-minimum-margin", type=float, default=0.05)
    parser.add_argument("--root-halving-transfer-fraction", type=float, default=0.35)
    parser.add_argument("--replay-split-namespace")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_ocak_sanity(
        OcakRunConfig(
            run_id=arguments.run_id,
            artifact_root=arguments.artifact_root,
            telemetry_path=arguments.telemetry,
            run_seed=arguments.seed,
            games=arguments.games,
            workers=arguments.workers,
            simulations=arguments.simulations,
            max_plies=arguments.max_plies,
            training_steps=arguments.training_steps,
            batch_size=arguments.batch_size,
            minimum_decisive_games=arguments.minimum_decisive_games,
            maximum_max_ply_draw_ratio=arguments.maximum_max_ply_draw_ratio,
            maximum_repetition_draw_ratio=arguments.maximum_repetition_draw_ratio,
            early_stopping_patience=arguments.early_stopping_patience,
            validation_interval_steps=arguments.validation_interval_steps,
            minimum_validation_delta=arguments.minimum_validation_delta,
            continuation_shards=tuple(arguments.continuation_shard),
            continuation_batch_fraction=arguments.continuation_batch_fraction,
            initial_model=arguments.initial_model,
            inference_wait_seconds=arguments.inference_wait_ms / 1_000,
            continuation_recency_decay=arguments.continuation_recency_decay,
            value_policy_temperature=arguments.value_policy_temperature,
            value_policy_prior_visits=arguments.value_policy_prior_visits,
            maximum_value_logit_adjustment=(
                arguments.maximum_value_logit_adjustment
            ),
            root_halving_enabled=arguments.root_halving,
            root_halving_top_actions=arguments.root_halving_top_actions,
            root_halving_finalists=arguments.root_halving_finalists,
            root_halving_first_round_simulations=(
                arguments.root_halving_first_round_simulations
            ),
            root_halving_final_round_simulations=(
                arguments.root_halving_final_round_simulations
            ),
            root_halving_minimum_margin=arguments.root_halving_minimum_margin,
            root_halving_transfer_fraction=arguments.root_halving_transfer_fraction,
            replay_split_namespace=arguments.replay_split_namespace,
        )
    )
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
