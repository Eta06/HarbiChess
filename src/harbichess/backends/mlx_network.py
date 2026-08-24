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

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (8, 8, self.config.input_channels):
            raise ValueError(
                "network input must have shape "
                f"(batch, 8, 8, {self.config.input_channels}), got {tuple(inputs.shape)}"
            )
        trunk = nn.relu(self.stem(inputs))
        for block in self.blocks:
            trunk = block(trunk)

        policy = nn.relu(self.policy_conv(trunk)).reshape(inputs.shape[0], -1)
        policy_logits = self.policy_linear(policy)
        value = nn.relu(self.value_conv(trunk)).reshape(inputs.shape[0], -1)
        value = nn.relu(self.value_hidden(value))
        wdl_logits = self.value_output(value)
        return policy_logits, wdl_logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.size for _, parameter in tree_flatten(self.parameters()))
