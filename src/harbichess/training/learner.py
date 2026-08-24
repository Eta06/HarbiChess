"""Guarded MLX policy/WDL learner for HarbiChess replay batches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.training.batch import TrainingBatch


class NonFiniteTrainingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_gradient_norm: float = 5.0
    policy_weight: float = 1.0
    value_weight: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.learning_rate,
            self.weight_decay,
            self.max_gradient_norm,
            self.policy_weight,
            self.value_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("learner configuration must be finite and non-negative")
        if self.learning_rate == 0 or self.max_gradient_norm == 0:
            raise ValueError("learning rate and max gradient norm must be positive")
        if self.policy_weight + self.value_weight == 0:
            raise ValueError("at least one learner loss weight must be positive")


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    step: int
    policy_loss: float
    value_loss: float
    total_loss: float
    gradient_norm: float


class MLXLearner:
    def __init__(
        self,
        network: HarbiChessNetwork,
        *,
        config: LearnerConfig | None = None,
        optimizer: optim.Optimizer | None = None,
    ) -> None:
        self.network = network
        self.config = config or LearnerConfig()
        self.optimizer = optimizer or optim.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.step = 0
        self._loss_and_grad = nn.value_and_grad(self.network, self._loss)

    def _loss(
        self,
        inputs: mx.array,
        policy_targets: mx.array,
        wdl_targets: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        policy_logits, wdl_logits = self.network(inputs)
        policy_loss = nn.losses.cross_entropy(
            policy_logits,
            policy_targets,
            reduction="mean",
        )
        value_loss = nn.losses.cross_entropy(
            wdl_logits,
            wdl_targets,
            reduction="mean",
        )
        total = (
            self.config.policy_weight * policy_loss
            + self.config.value_weight * value_loss
        )
        return total, policy_loss, value_loss

    @staticmethod
    def _arrays(batch: TrainingBatch) -> tuple[mx.array, mx.array, mx.array]:
        shape = batch.positions[0].shape
        if any(position.shape != shape for position in batch.positions):
            raise ValueError("training positions must share one encoded shape")
        inputs = mx.array([position.values for position in batch.positions], dtype=mx.float32)
        inputs = inputs.reshape((len(batch.positions), *shape))
        policies = mx.array(batch.policy_targets, dtype=mx.float32)
        wdl = mx.array(batch.wdl_targets, dtype=mx.int32)
        return inputs, policies, wdl

    @staticmethod
    def _tree_is_finite(tree: object) -> mx.array:
        checks = [mx.all(mx.isfinite(array)) for _, array in tree_flatten(tree)]
        if not checks:
            return mx.array(True)
        result = checks[0]
        for check in checks[1:]:
            result = mx.logical_and(result, check)
        return result

    def train_step(self, batch: TrainingBatch) -> TrainingMetrics:
        inputs, policies, wdl = self._arrays(batch)
        (total, policy_loss, value_loss), gradients = self._loss_and_grad(
            inputs,
            policies,
            wdl,
        )
        gradients, gradient_norm = optim.clip_grad_norm(
            gradients,
            self.config.max_gradient_norm,
        )
        gradients_finite = self._tree_is_finite(gradients)
        mx.eval(
            total,
            policy_loss,
            value_loss,
            gradient_norm,
            gradients_finite,
            gradients,
        )
        if not bool(gradients_finite.item()) or not all(
            math.isfinite(float(value.item()))
            for value in (total, policy_loss, value_loss, gradient_norm)
        ):
            raise NonFiniteTrainingError("non-finite loss or gradient; optimizer was not updated")
        self.optimizer.update(self.network, gradients)
        mx.eval(self.network.parameters(), self.optimizer.state)
        self.step += 1
        return TrainingMetrics(
            step=self.step,
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
            total_loss=float(total.item()),
            gradient_norm=float(gradient_norm.item()),
        )

    def evaluate_loss(self, batch: TrainingBatch) -> tuple[float, float, float]:
        inputs, policies, wdl = self._arrays(batch)
        total, policy_loss, value_loss = self._loss(inputs, policies, wdl)
        mx.eval(total, policy_loss, value_loss)
        return float(total.item()), float(policy_loss.item()), float(value_loss.item())
