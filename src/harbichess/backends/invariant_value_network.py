"""Policy-preserving network with independent global and spatial value residuals."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig, ResidualBlock
from harbichess.chess.encoding import HISTORY_STEPS, METADATA_PLANES, PIECE_PLANES_PER_STEP


@dataclass(frozen=True, slots=True)
class InvariantValueConfig:
    tower_channels: int = 16
    tower_blocks: int = 2
    tower_hidden: int = 32

    def __post_init__(self) -> None:
        if min(self.tower_channels, self.tower_blocks, self.tower_hidden) <= 0:
            raise ValueError("invariant value dimensions must be positive")


class HarbiChessInvariantValueNetwork(HarbiChessNetwork):
    """Keep the release path exact and add value-only residual representations."""

    def __init__(
        self,
        config: NetworkConfig | None = None,
        *,
        invariant_config: InvariantValueConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.invariant_config = invariant_config or InvariantValueConfig()
        self.invariant_value_linear = nn.Linear(
            PIECE_PLANES_PER_STEP + METADATA_PLANES, 3
        )
        self.value_tower_stem = nn.Conv2d(
            self.config.input_channels,
            self.invariant_config.tower_channels,
            kernel_size=3,
            padding=1,
        )
        self.value_tower_blocks = [
            ResidualBlock(self.invariant_config.tower_channels)
            for _ in range(self.invariant_config.tower_blocks)
        ]
        self.value_tower_hidden = nn.Linear(
            self.invariant_config.tower_channels * 2,
            self.invariant_config.tower_hidden,
        )
        self.value_tower_output = nn.Linear(self.invariant_config.tower_hidden, 3)
        self._zero_residual_outputs()

    def _zero_residual_outputs(self) -> None:
        self.invariant_value_linear.weight = mx.zeros_like(
            self.invariant_value_linear.weight
        )
        self.invariant_value_linear.bias = mx.zeros_like(self.invariant_value_linear.bias)
        self.value_tower_output.weight = mx.zeros_like(self.value_tower_output.weight)
        self.value_tower_output.bias = mx.zeros_like(self.value_tower_output.bias)

    @classmethod
    def from_base(
        cls,
        base: HarbiChessNetwork,
        *,
        invariant_config: InvariantValueConfig | None = None,
    ) -> HarbiChessInvariantValueNetwork:
        target = cls(base.config, invariant_config=invariant_config)
        target.load_weights(list(tree_flatten(base.parameters())), strict=False)
        target._zero_residual_outputs()
        mx.eval(target.parameters())
        return target

    def _value_residual(self, inputs: mx.array) -> mx.array:
        current_piece_counts = mx.sum(
            inputs[:, :, :, :PIECE_PLANES_PER_STEP], axis=(1, 2)
        )
        metadata_start = HISTORY_STEPS * PIECE_PLANES_PER_STEP
        metadata = mx.mean(inputs[:, :, :, metadata_start:], axis=(1, 2))
        invariant_features = mx.concatenate(
            (current_piece_counts, metadata), axis=1
        )
        invariant = self.invariant_value_linear(invariant_features)
        tower = nn.relu(self.value_tower_stem(inputs))
        for block in self.value_tower_blocks:
            tower = block(tower)
        pooled = mx.concatenate(
            (mx.mean(tower, axis=(1, 2)), mx.max(tower, axis=(1, 2))), axis=1
        )
        tower_hidden = nn.relu(self.value_tower_hidden(pooled))
        return invariant + self.value_tower_output(tower_hidden)

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        trunk = self._trunk(inputs)
        policy_logits = self.policy_linear(self._policy_features(trunk))
        value_logits = self._value_logits(trunk) + self._value_residual(inputs)
        return policy_logits, value_logits

    def masked_policy_value(
        self,
        inputs: mx.array,
        action_indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if (
            action_indices.ndim != 2
            or action_indices.shape[0] != inputs.shape[0]
            or action_indices.shape[1] == 0
        ):
            raise ValueError("masked actions must have shape (batch, non-zero actions)")
        trunk = self._trunk(inputs)
        policy = self._policy_features(trunk)
        flat_actions = action_indices.reshape(-1)
        selected_weights = mx.take(
            self.policy_linear.weight, flat_actions, axis=0
        ).reshape(
            action_indices.shape[0],
            action_indices.shape[1],
            policy.shape[1],
        )
        selected_bias = mx.take(
            self.policy_linear.bias, flat_actions, axis=0
        ).reshape(action_indices.shape)
        policy_logits = (
            mx.sum(policy[:, None, :] * selected_weights, axis=2) + selected_bias
        )
        value_logits = self._value_logits(trunk) + self._value_residual(inputs)
        return policy_logits, value_logits

    def freeze_release_parameters(self) -> None:
        """Leave only the two new value residual branches trainable."""

        self.freeze()
        self.invariant_value_linear.unfreeze()
        self.value_tower_stem.unfreeze()
        for block in self.value_tower_blocks:
            block.unfreeze()
        self.value_tower_hidden.unfreeze()
        self.value_tower_output.unfreeze()

    def freeze_to_global_linear(self) -> None:
        """Leave only the direct invariant projection trainable."""

        self.freeze()
        self.invariant_value_linear.unfreeze()
