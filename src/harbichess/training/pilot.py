"""Small gated learner pilot that must pass before any large training run."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from harbichess.replay.schema import ReplayRecord
from harbichess.training.batch import GameBalancedSampler, TrainingBatch, build_training_batch
from harbichess.training.learner import LearnerSnapshot, MLXLearner, TrainingMetrics


class PilotStopReason(StrEnum):
    MAX_STEPS = "max_steps"
    EARLY_STOPPING = "early_stopping_no_validation_improvement"


@dataclass(frozen=True, slots=True)
class PilotConfig:
    steps: int = 100
    batch_size: int = 32
    minimum_train_improvement: float = 0.05
    maximum_validation_ratio: float = 1.25
    validation_interval_steps: int = 10
    early_stopping_patience: int = 5
    minimum_validation_delta: float = 1e-3
    maximum_value_validation_ratio: float = 1.05
    checkpoint_interval_steps: int = 8
    maximum_validation_checkpoints: int = 4
    continuation_fraction: float | None = None
    continuation_game_weights: Mapping[str, float] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            self.steps <= 0
            or self.batch_size <= 0
            or self.validation_interval_steps <= 0
            or self.early_stopping_patience <= 0
            or self.checkpoint_interval_steps <= 0
            or self.maximum_validation_checkpoints <= 0
            or not math.isfinite(self.maximum_value_validation_ratio)
            or self.maximum_value_validation_ratio < 1.0
        ):
            raise ValueError("pilot steps and batch size must be positive")
        if not 0.0 <= self.minimum_train_improvement < 1.0:
            raise ValueError("minimum train improvement must be in [0, 1)")
        if self.maximum_validation_ratio <= 0:
            raise ValueError("maximum validation ratio must be positive")
        if not math.isfinite(self.minimum_validation_delta) or self.minimum_validation_delta < 0:
            raise ValueError("minimum validation delta must be finite and non-negative")
        if self.continuation_fraction is not None and not 0.0 <= self.continuation_fraction <= 1.0:
            raise ValueError("continuation fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PilotReport:
    passed: bool
    reasons: tuple[str, ...]
    steps: int
    initial_train_loss: float
    final_train_loss: float
    initial_validation_loss: float
    final_validation_loss: float
    initial_validation_value_loss: float
    final_validation_value_loss: float
    maximum_gradient_norm: float
    maximum_unclipped_gradient_norm: float
    metrics: tuple[TrainingMetrics, ...]
    sampler_rng_state: object
    attempted_steps: int
    best_validation_step: int
    best_validation_loss: float
    best_validation_value_loss: float
    stopped_early: bool
    validation_candidates: tuple[ValidationCandidate, ...]
    stop_reason: PilotStopReason
    last_validation_step: int
    last_validation_loss: float
    last_validation_value_loss: float
    last_improvement_step: int
    stale_validation_evaluations: int
    validation_evaluations: int
    train_value_samples: int
    validation_value_samples: int


@dataclass(frozen=True, slots=True)
class ValidationCandidate:
    step: int
    validation_loss: float
    validation_policy_loss: float
    validation_value_loss: float
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
    raw_train_eval = train_evaluation or build_training_batch(train_records)
    raw_validation_eval = validation_evaluation or build_training_batch(validation_records)
    if len(raw_train_eval.positions) != len(train_records):
        raise ValueError("train evaluation batch does not match replay records")
    if len(raw_validation_eval.positions) != len(validation_records):
        raise ValueError("validation evaluation batch does not match replay records")
    train_value_samples = sum(weight > 0 for weight in raw_train_eval.value_weights)
    validation_value_samples = sum(weight > 0 for weight in raw_validation_eval.value_weights)
    train_eval = learner.prepare_batch(raw_train_eval)
    validation_eval = learner.prepare_batch(raw_validation_eval)
    initial_train = learner.evaluate_loss(train_eval)[0]
    initial_validation, _, initial_validation_value = learner.evaluate_loss(validation_eval)
    sampler = GameBalancedSampler(
        train_records,
        seed=settings.seed,
        continuation_fraction=settings.continuation_fraction,
        continuation_game_weights=settings.continuation_game_weights,
    )
    best_snapshot = learner.snapshot()
    best_sampler_state = sampler.rng_state
    best_validation = initial_validation
    best_validation_value = initial_validation_value
    best_validation_step = learner.step
    stale_evaluations = 0
    stopped_early = False
    stop_reason = PilotStopReason.MAX_STEPS
    last_validation_step = learner.step
    last_validation_loss = initial_validation
    last_validation_value_loss = initial_validation_value
    last_improvement_step = learner.step
    validation_evaluations = 0
    validation_candidates: list[ValidationCandidate] = []
    metrics = []
    for _ in range(settings.steps):
        sampled_indices = sampler.sample_indices(settings.batch_size)
        metric = learner.train_step(train_eval.select(sampled_indices))
        metrics.append(metric)
        validation_loss = None
        if metric.step % settings.validation_interval_steps == 0 or metric.step == settings.steps:
            validation_loss, validation_policy_loss, validation_value_loss = learner.evaluate_loss(
                validation_eval
            )
            validation_evaluations += 1
            last_validation_step = metric.step
            last_validation_loss = validation_loss
            last_validation_value_loss = validation_value_loss
            value_safe = (
                validation_value_samples > 0
                and validation_value_loss
                <= initial_validation_value * settings.maximum_value_validation_ratio
            )
            if validation_loss < best_validation - settings.minimum_validation_delta and value_safe:
                best_validation = validation_loss
                best_validation_value = validation_value_loss
                best_validation_step = metric.step
                best_snapshot = learner.snapshot()
                best_sampler_state = sampler.rng_state
                last_improvement_step = metric.step
                candidate = ValidationCandidate(
                    step=metric.step,
                    validation_loss=validation_loss,
                    validation_policy_loss=validation_policy_loss,
                    validation_value_loss=validation_value_loss,
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
            stop_reason = PilotStopReason.EARLY_STOPPING
            break
    attempted_steps = learner.step
    learner.restore(best_snapshot)
    sampler.set_rng_state(best_sampler_state)
    final_train = learner.evaluate_loss(train_eval)[0]
    final_validation, _, final_validation_value = learner.evaluate_loss(validation_eval)

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
    if train_value_samples == 0:
        reasons.append("training replay has no known value targets")
    if validation_value_samples == 0:
        reasons.append("validation replay has no known value targets")
    return PilotReport(
        passed=not reasons,
        reasons=tuple(reasons),
        steps=learner.step,
        initial_train_loss=initial_train,
        final_train_loss=final_train,
        initial_validation_loss=initial_validation,
        final_validation_loss=final_validation,
        initial_validation_value_loss=initial_validation_value,
        final_validation_value_loss=final_validation_value,
        maximum_gradient_norm=max(metric.gradient_norm for metric in metrics),
        maximum_unclipped_gradient_norm=max(
            metric.unclipped_gradient_norm for metric in metrics
        ),
        metrics=tuple(metrics),
        sampler_rng_state=sampler.rng_state,
        attempted_steps=attempted_steps,
        best_validation_step=best_validation_step,
        best_validation_loss=best_validation,
        best_validation_value_loss=best_validation_value,
        stopped_early=stopped_early,
        validation_candidates=tuple(validation_candidates),
        stop_reason=stop_reason,
        last_validation_step=last_validation_step,
        last_validation_loss=last_validation_loss,
        last_validation_value_loss=last_validation_value_loss,
        last_improvement_step=last_improvement_step,
        stale_validation_evaluations=stale_evaluations,
        validation_evaluations=validation_evaluations,
        train_value_samples=train_value_samples,
        validation_value_samples=validation_value_samples,
    )
