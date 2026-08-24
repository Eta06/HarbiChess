"""Small gated learner pilot that must pass before any large training run."""

from __future__ import annotations

import math
from dataclasses import dataclass

from harbichess.replay.schema import ReplayRecord
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.learner import MLXLearner, TrainingMetrics


@dataclass(frozen=True, slots=True)
class PilotConfig:
    steps: int = 100
    batch_size: int = 32
    minimum_train_improvement: float = 0.05
    maximum_validation_ratio: float = 1.25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("pilot steps and batch size must be positive")
        if not 0.0 <= self.minimum_train_improvement < 1.0:
            raise ValueError("minimum train improvement must be in [0, 1)")
        if self.maximum_validation_ratio <= 0:
            raise ValueError("maximum validation ratio must be positive")


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


def run_sanity_pilot(
    learner: MLXLearner,
    train_records: tuple[ReplayRecord, ...],
    validation_records: tuple[ReplayRecord, ...],
    *,
    config: PilotConfig | None = None,
) -> PilotReport:
    settings = config or PilotConfig()
    if not train_records or not validation_records:
        raise ValueError("pilot requires non-empty train and validation records")
    train_games = {record.game_id for record in train_records}
    validation_games = {record.game_id for record in validation_records}
    if train_games & validation_games:
        raise ValueError("train and validation records leak the same game IDs")

    train_eval = build_training_batch(train_records)
    validation_eval = build_training_batch(validation_records)
    initial_train = learner.evaluate_loss(train_eval)[0]
    initial_validation = learner.evaluate_loss(validation_eval)[0]
    sampler = GameBalancedSampler(train_records, seed=settings.seed)
    metrics = []
    for _ in range(settings.steps):
        sampled = sampler.sample(settings.batch_size)
        metrics.append(learner.train_step(build_training_batch(sampled)))
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
        steps=settings.steps,
        initial_train_loss=initial_train,
        final_train_loss=final_train,
        initial_validation_loss=initial_validation,
        final_validation_loss=final_validation,
        maximum_gradient_norm=max(metric.gradient_norm for metric in metrics),
        metrics=tuple(metrics),
    )
