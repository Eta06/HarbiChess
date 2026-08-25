"""Small gated learner pilot that must pass before any large training run."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from harbichess.replay.schema import ReplayRecord
from harbichess.training.batch import GameBalancedSampler, TrainingBatch, build_training_batch
from harbichess.training.learner import LearnerSnapshot, MLXLearner, TrainingMetrics


@dataclass(frozen=True, slots=True)
class PilotConfig:
    steps: int = 100
    batch_size: int = 32
    minimum_train_improvement: float = 0.05
    maximum_validation_ratio: float = 1.25
    validation_interval_steps: int = 10
    early_stopping_patience: int = 5
    minimum_validation_delta: float = 1e-3
    checkpoint_interval_steps: int = 8
    maximum_validation_checkpoints: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            self.steps <= 0
            or self.batch_size <= 0
            or self.validation_interval_steps <= 0
            or self.early_stopping_patience <= 0
            or self.checkpoint_interval_steps <= 0
            or self.maximum_validation_checkpoints <= 0
        ):
            raise ValueError("pilot steps and batch size must be positive")
        if not 0.0 <= self.minimum_train_improvement < 1.0:
            raise ValueError("minimum train improvement must be in [0, 1)")
        if self.maximum_validation_ratio <= 0:
            raise ValueError("maximum validation ratio must be positive")
        if not math.isfinite(self.minimum_validation_delta) or self.minimum_validation_delta < 0:
            raise ValueError("minimum validation delta must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PilotReport:
    passed: bool
    reasons: tuple[str, ...]
    steps: int
    initial_train_loss: float
    final_train_loss: float
    initial_validation_loss: float
    final_validation_loss: float
    maximum_gradient_norm: float
    metrics: tuple[TrainingMetrics, ...]
    sampler_rng_state: object
    attempted_steps: int
    best_validation_step: int
    best_validation_loss: float
    stopped_early: bool
    validation_candidates: tuple[ValidationCandidate, ...]


@dataclass(frozen=True, slots=True)
class ValidationCandidate:
    step: int
    validation_loss: float
    learner_snapshot: LearnerSnapshot
    sampler_rng_state: object


def run_sanity_pilot(
    learner: MLXLearner,
    train_records: tuple[ReplayRecord, ...],
    validation_records: tuple[ReplayRecord, ...],
    *,
    config: PilotConfig | None = None,
    on_step: Callable[[TrainingMetrics, float | None], None] | None = None,
    train_evaluation: TrainingBatch | None = None,
    validation_evaluation: TrainingBatch | None = None,
) -> PilotReport:
    settings = config or PilotConfig()
    if not train_records or not validation_records:
        raise ValueError("pilot requires non-empty train and validation records")
    train_games = {record.game_id for record in train_records}
    validation_games = {record.game_id for record in validation_records}
    if train_games & validation_games:
        raise ValueError("train and validation records leak the same game IDs")

    if (train_evaluation is None) != (validation_evaluation is None):
        raise ValueError("train and validation evaluation batches must be supplied together")
    train_eval = train_evaluation or build_training_batch(train_records)
    validation_eval = validation_evaluation or build_training_batch(validation_records)
    if len(train_eval.positions) != len(train_records):
        raise ValueError("train evaluation batch does not match replay records")
    if len(validation_eval.positions) != len(validation_records):
        raise ValueError("validation evaluation batch does not match replay records")
    initial_train = learner.evaluate_loss(train_eval)[0]
    initial_validation = learner.evaluate_loss(validation_eval)[0]
    sampler = GameBalancedSampler(train_records, seed=settings.seed)
    best_snapshot = learner.snapshot()
    best_sampler_state = sampler.rng_state
    best_validation = initial_validation
    best_validation_step = learner.step
    stale_evaluations = 0
    stopped_early = False
    validation_candidates: list[ValidationCandidate] = []
    metrics = []
    for _ in range(settings.steps):
        sampled_indices = sampler.sample_indices(settings.batch_size)
        metric = learner.train_step(train_eval.select(sampled_indices))
        metrics.append(metric)
        validation_loss = None
        if metric.step % settings.validation_interval_steps == 0 or metric.step == settings.steps:
            validation_loss = learner.evaluate_loss(validation_eval)[0]
            if validation_loss < best_validation - settings.minimum_validation_delta:
                best_validation = validation_loss
                best_validation_step = metric.step
                best_snapshot = learner.snapshot()
                best_sampler_state = sampler.rng_state
                candidate = ValidationCandidate(
                    step=metric.step,
                    validation_loss=validation_loss,
                    learner_snapshot=best_snapshot,
                    sampler_rng_state=best_sampler_state,
                )
                if (
                    validation_candidates
                    and metric.step - validation_candidates[-1].step
                    < settings.checkpoint_interval_steps
                ):
                    validation_candidates[-1] = candidate
                else:
                    validation_candidates.append(candidate)
                validation_candidates = validation_candidates[
                    -settings.maximum_validation_checkpoints :
                ]
                stale_evaluations = 0
            else:
                stale_evaluations += 1
        if on_step is not None:
            on_step(metric, validation_loss)
        if stale_evaluations >= settings.early_stopping_patience:
            stopped_early = True
            break
    attempted_steps = learner.step
    learner.restore(best_snapshot)
    sampler.set_rng_state(best_sampler_state)
    final_train = learner.evaluate_loss(train_eval)[0]
    final_validation = learner.evaluate_loss(validation_eval)[0]

    reasons = []
    if not all(
        math.isfinite(value)
        for value in (initial_train, initial_validation, final_train, final_validation)
    ):
        reasons.append("losses are not finite")
    if final_train > initial_train * (1.0 - settings.minimum_train_improvement):
        reasons.append("training loss did not improve enough")
    if final_validation > initial_validation * settings.maximum_validation_ratio:
        reasons.append("validation loss degraded beyond the safety ratio")
    return PilotReport(
        passed=not reasons,
        reasons=tuple(reasons),
        steps=learner.step,
        initial_train_loss=initial_train,
        final_train_loss=final_train,
        initial_validation_loss=initial_validation,
        final_validation_loss=final_validation,
        maximum_gradient_norm=max(metric.gradient_norm for metric in metrics),
        metrics=tuple(metrics),
        sampler_rng_state=sampler.rng_state,
        attempted_steps=attempted_steps,
        best_validation_step=best_validation_step,
        best_validation_loss=best_validation,
        stopped_early=stopped_early,
        validation_candidates=tuple(validation_candidates),
    )
