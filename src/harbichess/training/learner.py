"""Guarded MLX policy/WDL learner for HarbiChess replay batches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

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


@dataclass(frozen=True, slots=True)
class LearnerSnapshot:
    step: int
    model_weights: tuple[tuple[str, mx.array], ...]
    optimizer_state: tuple[tuple[str, mx.array], ...]


@dataclass(frozen=True, slots=True)
class PreparedTrainingBatch:
    inputs: mx.array
    policy_targets: mx.array
    wdl_targets: mx.array
    value_weights: mx.array

    @property
    def size(self) -> int:
        return self.inputs.shape[0]

    def select(self, indices: tuple[int, ...]) -> PreparedTrainingBatch:
        if not indices or any(index < 0 or index >= self.size for index in indices):
            raise IndexError("prepared batch indices must be non-empty and in range")
        rows = mx.array(indices, dtype=mx.int32)
        return PreparedTrainingBatch(
            mx.take(self.inputs, rows, axis=0),
            mx.take(self.policy_targets, rows, axis=0),
            mx.take(self.wdl_targets, rows, axis=0),
            mx.take(self.value_weights, rows, axis=0),
        )


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
        value_weights: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        policy_logits, wdl_logits = self.network(inputs)
        policy_loss = nn.losses.cross_entropy(
            policy_logits,
            policy_targets,
            reduction="mean",
        )
        value_losses = nn.losses.cross_entropy(
            wdl_logits,
            wdl_targets,
            reduction="none",
        )
        value_loss = mx.sum(value_losses * value_weights) / mx.maximum(
            mx.sum(value_weights),
            mx.array(1.0),
        )
        total = self.config.policy_weight * policy_loss + self.config.value_weight * value_loss
        return total, policy_loss, value_loss

    @staticmethod
    def _arrays(batch: TrainingBatch) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        shape = batch.positions[0].shape
        if any(position.shape != shape for position in batch.positions):
            raise ValueError("training positions must share one encoded shape")
        inputs = mx.array([position.values for position in batch.positions], dtype=mx.float32)
        inputs = inputs.reshape((len(batch.positions), *shape))
        policies = mx.array(batch.policy_targets, dtype=mx.float32)
        wdl = mx.array(batch.wdl_targets, dtype=mx.int32)
        value_weights = mx.array(batch.value_weights, dtype=mx.float32)
        return inputs, policies, wdl, value_weights

    @classmethod
    def prepare_batch(cls, batch: TrainingBatch) -> PreparedTrainingBatch:
        prepared = PreparedTrainingBatch(*cls._arrays(batch))
        mx.eval(
            prepared.inputs,
            prepared.policy_targets,
            prepared.wdl_targets,
            prepared.value_weights,
        )
        return prepared

    @staticmethod
    def _prepared_arrays(
        batch: TrainingBatch | PreparedTrainingBatch,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if isinstance(batch, PreparedTrainingBatch):
            return (
                batch.inputs,
                batch.policy_targets,
                batch.wdl_targets,
                batch.value_weights,
            )
        return MLXLearner._arrays(batch)

    @staticmethod
    def _tree_is_finite(tree: object) -> mx.array:
        checks = [mx.all(mx.isfinite(array)) for _, array in tree_flatten(tree)]
        if not checks:
            return mx.array(True)
        result = checks[0]
        for check in checks[1:]:
            result = mx.logical_and(result, check)
        return result

    def train_step(
        self,
        batch: TrainingBatch | PreparedTrainingBatch,
    ) -> TrainingMetrics:
        inputs, policies, wdl, value_weights = self._prepared_arrays(batch)
        (total, policy_loss, value_loss), gradients = self._loss_and_grad(
            inputs,
            policies,
            wdl,
            value_weights,
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

    def evaluate_loss(
        self,
        batch: TrainingBatch | PreparedTrainingBatch,
    ) -> tuple[float, float, float]:
        inputs, policies, wdl, value_weights = self._prepared_arrays(batch)
        total, policy_loss, value_loss = self._loss(inputs, policies, wdl, value_weights)
        mx.eval(total, policy_loss, value_loss)
        return float(total.item()), float(policy_loss.item()), float(value_loss.item())

    def snapshot(self) -> LearnerSnapshot:
        model_weights = tuple(
            (name, mx.array(array)) for name, array in tree_flatten(self.network.parameters())
        )
        optimizer_state = tuple(
            (name, mx.array(array)) for name, array in tree_flatten(self.optimizer.state)
        )
        mx.eval(
            [array for _, array in model_weights],
            [array for _, array in optimizer_state],
        )
        return LearnerSnapshot(self.step, model_weights, optimizer_state)

    def restore(self, snapshot: LearnerSnapshot) -> None:
        self.network.load_weights(list(snapshot.model_weights))
        self.optimizer.state = tree_unflatten(list(snapshot.optimizer_state))
        self.step = snapshot.step
        mx.eval(self.network.parameters(), self.optimizer.state)
