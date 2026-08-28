"""MLX residual policy/WDL network for HarbiChess."""

from __future__ import annotations

from dataclasses import dataclass, fields

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from harbichess.chess.actions import POLICY_SIZE
from harbichess.chess.encoding import ENCODER_CHANNELS


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    input_channels: int = ENCODER_CHANNELS
    trunk_channels: int = 64
    residual_blocks: int = 4
    policy_channels: int = 8
    value_channels: int = 4
    value_hidden: int = 64
    policy_size: int = POLICY_SIZE

    def __post_init__(self) -> None:
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden = nn.relu(self.conv1(inputs))
        return nn.relu(inputs + self.conv2(hidden))


class HarbiChessNetwork(nn.Module):
    """A compact residual CNN with global policy and WDL heads."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.stem = nn.Conv2d(
            self.config.input_channels,
            self.config.trunk_channels,
            kernel_size=3,
            padding=1,
        )
        self.blocks = [
            ResidualBlock(self.config.trunk_channels)
            for _ in range(self.config.residual_blocks)
        ]
        self.policy_conv = nn.Conv2d(
            self.config.trunk_channels,
            self.config.policy_channels,
            kernel_size=1,
        )
        self.policy_linear = nn.Linear(
            8 * 8 * self.config.policy_channels,
            self.config.policy_size,
        )
        self.value_conv = nn.Conv2d(
            self.config.trunk_channels,
            self.config.value_channels,
            kernel_size=1,
        )
        self.value_hidden = nn.Linear(8 * 8 * self.config.value_channels, self.config.value_hidden)
        self.value_output = nn.Linear(self.config.value_hidden, 3)

    def _validate_inputs(self, inputs: mx.array) -> None:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (8, 8, self.config.input_channels):
            raise ValueError(
                "network input must have shape "
                f"(batch, 8, 8, {self.config.input_channels}), got {tuple(inputs.shape)}"
            )

    def _trunk(self, inputs: mx.array) -> mx.array:
        self._validate_inputs(inputs)
        trunk = nn.relu(self.stem(inputs))
        for block in self.blocks:
            trunk = block(trunk)
        return trunk

    def _policy_features(self, trunk: mx.array) -> mx.array:
        return nn.relu(self.policy_conv(trunk)).reshape(trunk.shape[0], -1)

    def _value_logits(self, trunk: mx.array) -> mx.array:
        value = nn.relu(self.value_conv(trunk)).reshape(trunk.shape[0], -1)
        value = nn.relu(self.value_hidden(value))
        return self.value_output(value)

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        trunk = self._trunk(inputs)
        policy_logits = self.policy_linear(self._policy_features(trunk))
        return policy_logits, self._value_logits(trunk)

    def masked_policy_value(
        self,
        inputs: mx.array,
        action_indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Evaluate only requested policy logits while retaining the exact WDL head."""

        if (
            action_indices.ndim != 2
            or action_indices.shape[0] != inputs.shape[0]
            or action_indices.shape[1] == 0
        ):
            raise ValueError("masked actions must have shape (batch, non-zero actions)")
        trunk = self._trunk(inputs)
        policy = self._policy_features(trunk)
        flat_actions = action_indices.reshape(-1)
        selected_weights = mx.take(self.policy_linear.weight, flat_actions, axis=0).reshape(
            action_indices.shape[0],
            action_indices.shape[1],
            policy.shape[1],
        )
        selected_bias = mx.take(self.policy_linear.bias, flat_actions, axis=0).reshape(
            action_indices.shape
        )
        policy_logits = mx.sum(policy[:, None, :] * selected_weights, axis=2) + selected_bias
        return policy_logits, self._value_logits(trunk)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.size for _, parameter in tree_flatten(self.parameters()))
