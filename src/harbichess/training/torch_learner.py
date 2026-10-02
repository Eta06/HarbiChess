"""Real FP32 policy/WDL optimization with the same losses as the MLX learner."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from harbichess.backends.torch_network import TorchChessNetwork
from harbichess.chess.encoding import ENCODER_SCHEMA_VERSION
from harbichess.training.batch import TrainingBatch
from harbichess.training.config import LearnerConfig, NonFiniteTrainingError, TrainingMetrics


@dataclass(frozen=True)
class TorchTrainingBatch:
    inputs: torch.Tensor
    policies: torch.Tensor
    legal_masks: torch.Tensor
    wdl: torch.Tensor
    value_weights: torch.Tensor

    @property
    def size(self) -> int:
        return self.inputs.shape[0]

    def select(self, indices: tuple[int, ...]) -> TorchTrainingBatch:
        if not indices or any(i < 0 or i >= self.size for i in indices):
            raise IndexError("prepared batch indices must be non-empty and in range")
        rows = torch.tensor(indices, device=self.inputs.device, dtype=torch.long)
        return TorchTrainingBatch(
            *(
                x.index_select(0, rows)
                for x in (
                    self.inputs,
                    self.policies,
                    self.legal_masks,
                    self.wdl,
                    self.value_weights,
                )
            )
        )


class TorchLearner:
    def __init__(
        self,
        network: TorchChessNetwork,
        *,
        config: LearnerConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.network = network.to(device=device, dtype=torch.float32).train()
        self.device = torch.device(device)
        self.config = config or LearnerConfig()
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.step = 0

    def prepare_batch(self, batch: TrainingBatch) -> TorchTrainingBatch:
        shape = (8, 8, self.network.config.input_channels)
        if any(
            p.shape != shape or p.schema_version != ENCODER_SCHEMA_VERSION for p in batch.positions
        ):
            raise ValueError("training shape/schema incompatible with network")

        def tensor(x, dtype):
            return torch.tensor(x, dtype=dtype, device=self.device)

        return TorchTrainingBatch(
            tensor([p.values for p in batch.positions], torch.float32).reshape(-1, *shape),
            tensor(batch.policy_targets, torch.float32),
            tensor(batch.legal_masks, torch.bool),
            tensor(batch.wdl_targets, torch.long),
            tensor(batch.value_weights, torch.float32),
        )

    def _loss(self, batch: TorchTrainingBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy, wdl = self.network(batch.inputs)
        masked = policy.masked_fill(~batch.legal_masks, -1e9)
        log_policy = F.log_softmax(masked, dim=1)
        policy_loss = -(batch.policies * log_policy).sum(1).mean()
        if self.config.confident_top_action_weight:
            top = batch.policies.topk(2, dim=1).values
            weights = (top[:, 0] - top[:, 1] >= self.config.confident_top_action_margin).float()
            losses = F.cross_entropy(masked, batch.policies.argmax(1), reduction="none")
            policy_loss = policy_loss + self.config.confident_top_action_weight * (
                (losses * weights).sum() / weights.sum().clamp_min(1)
            )
        value_losses = F.cross_entropy(wdl, batch.wdl, reduction="none")
        value_loss = (
            value_losses * batch.value_weights
        ).sum() / batch.value_weights.sum().clamp_min(1)
        return (
            self.config.policy_weight * policy_loss + self.config.value_weight * value_loss,
            policy_loss,
            value_loss,
        )

    def evaluate_loss(self, batch: TorchTrainingBatch) -> tuple[float, float, float]:
        with torch.no_grad():
            return tuple(float(x) for x in self._loss(batch))

    def train_step(self, batch: TorchTrainingBatch) -> TrainingMetrics:
        self.optimizer.zero_grad(set_to_none=True)
        total, policy, value = self._loss(batch)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), self.config.max_gradient_norm, error_if_nonfinite=False
        )
        losses = tuple(float(x.detach()) for x in (total, policy, value, norm))
        if not all(math.isfinite(x) for x in losses) or any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in self.network.parameters()
        ):
            self.optimizer.zero_grad(set_to_none=True)
            raise NonFiniteTrainingError("non-finite loss or gradient; optimizer was not updated")
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.network.parameters()):
            raise NonFiniteTrainingError(
                "non-finite optimizer update; stop and restore last checkpoint"
            )
        self.step += 1
        return TrainingMetrics(
            self.step,
            losses[1],
            losses[2],
            losses[0],
            min(losses[3], self.config.max_gradient_norm),
            losses[3],
        )
