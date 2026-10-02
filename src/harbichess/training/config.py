"""Framework-neutral learner loss, optimization and telemetry contracts."""

import math
from dataclasses import dataclass


class NonFiniteTrainingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_gradient_norm: float = 5.0
    policy_weight: float = 1.0
    value_weight: float = 1.0
    confident_top_action_weight: float = 0.0
    confident_top_action_margin: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.learning_rate,
            self.weight_decay,
            self.max_gradient_norm,
            self.policy_weight,
            self.value_weight,
            self.confident_top_action_weight,
            self.confident_top_action_margin,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("learner configuration must be finite and non-negative")
        if self.learning_rate == 0 or self.max_gradient_norm == 0:
            raise ValueError("learning rate and max gradient norm must be positive")
        if self.policy_weight + self.value_weight == 0:
            raise ValueError("at least one learner loss weight must be positive")
        if self.confident_top_action_margin > 1.0:
            raise ValueError("confident top-action margin must not exceed one")


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    step: int
    policy_loss: float
    value_loss: float
    total_loss: float
    gradient_norm: float
    unclipped_gradient_norm: float
