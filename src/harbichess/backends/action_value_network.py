"""Function-preserving action-value representation for search improvement."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig


class ActionValueHead(nn.Module):
    """Dueling action advantages anchored to the existing WDL state value."""

    def __init__(self, trunk_channels: int, action_channels: int, policy_size: int) -> None:
        super().__init__()
        if min(trunk_channels, action_channels, policy_size) <= 0:
            raise ValueError("action-value head dimensions must be positive")
        self.conv = nn.Conv2d(trunk_channels, action_channels, kernel_size=1)
        self.output = nn.Linear(8 * 8 * action_channels, policy_size)
        self.output.weight = mx.zeros_like(self.output.weight)
        self.output.bias = mx.zeros_like(self.output.bias)

    def __call__(self, trunk: mx.array, state_value: mx.array) -> mx.array:
        if trunk.ndim != 4 or state_value.ndim != 1 or trunk.shape[0] != state_value.shape[0]:
            raise ValueError("action-value head requires batched trunk features and state values")
        features = nn.relu(self.conv(trunk)).reshape(trunk.shape[0], -1)
        advantages = self.output(features)
        return mx.tanh(state_value[:, None] + advantages)


class HarbiChessActionValueNetwork(HarbiChessNetwork):
    """HarbiChess network with a non-invasive legal-action Q head."""

    def __init__(
        self,
        config: NetworkConfig | None = None,
        *,
        action_value_channels: int = 4,
    ) -> None:
        super().__init__(config)
        self.action_value_channels = action_value_channels
        self.action_value_head = ActionValueHead(
            self.config.trunk_channels,
            action_value_channels,
            self.config.policy_size,
        )

    @classmethod
    def from_base(
        cls,
        base: HarbiChessNetwork,
        *,
        action_value_channels: int = 4,
    ) -> HarbiChessActionValueNetwork:
        target = cls(base.config, action_value_channels=action_value_channels)
        target.load_weights(list(tree_flatten(base.parameters())), strict=False)
        mx.eval(target.parameters())
        return target

    @staticmethod
    def expected_wdl_value(wdl_logits: mx.array) -> mx.array:
        probabilities = mx.softmax(wdl_logits, axis=1)
        return probabilities[:, 0] - probabilities[:, 2]

    def frozen_action_features(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        trunk = self._trunk(inputs)
        wdl_logits = self._value_logits(trunk)
        return mx.stop_gradient(trunk), mx.stop_gradient(self.expected_wdl_value(wdl_logits))

    def action_values(self, inputs: mx.array) -> mx.array:
        trunk = self._trunk(inputs)
        state_value = self.expected_wdl_value(self._value_logits(trunk))
        return self.action_value_head(trunk, state_value)

    def forward_with_action_values(
        self, inputs: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        trunk = self._trunk(inputs)
        policy_logits = self.policy_linear(self._policy_features(trunk))
        wdl_logits = self._value_logits(trunk)
        action_values = self.action_value_head(trunk, self.expected_wdl_value(wdl_logits))
        return policy_logits, wdl_logits, action_values


class SpatialActionValueHead(nn.Module):
    """Square-shared action advantages aligned with the 8x8x73 action schema."""

    def __init__(self, trunk_channels: int, action_planes: int = 73) -> None:
        super().__init__()
        if min(trunk_channels, action_planes) <= 0:
            raise ValueError("spatial action-value dimensions must be positive")
        self.output = nn.Conv2d(trunk_channels, action_planes, kernel_size=1)
        self.output.weight = mx.zeros_like(self.output.weight)
        self.output.bias = mx.zeros_like(self.output.bias)

    def __call__(self, trunk: mx.array, state_value: mx.array) -> mx.array:
        if trunk.ndim != 4 or state_value.ndim != 1 or trunk.shape[0] != state_value.shape[0]:
            raise ValueError("spatial action-value head requires batched features and values")
        advantages = self.output(trunk).reshape(trunk.shape[0], -1)
        return mx.tanh(state_value[:, None] + advantages)


class HarbiChessSpatialActionValueNetwork(HarbiChessNetwork):
    """HarbiChess network with a parameter-shared spatial legal-action Q head."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__(config)
        self.action_value_head = SpatialActionValueHead(self.config.trunk_channels)

    @classmethod
    def from_base(cls, base: HarbiChessNetwork) -> HarbiChessSpatialActionValueNetwork:
        target = cls(base.config)
        target.load_weights(list(tree_flatten(base.parameters())), strict=False)
        mx.eval(target.parameters())
        return target

    def frozen_action_features(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        trunk = self._trunk(inputs)
        wdl_logits = self._value_logits(trunk)
        state_value = HarbiChessActionValueNetwork.expected_wdl_value(wdl_logits)
        return mx.stop_gradient(trunk), mx.stop_gradient(state_value)

    def action_values(self, inputs: mx.array) -> mx.array:
        trunk = self._trunk(inputs)
        wdl_logits = self._value_logits(trunk)
        state_value = HarbiChessActionValueNetwork.expected_wdl_value(wdl_logits)
        return self.action_value_head(trunk, state_value)

    def forward_with_action_values(
        self, inputs: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        trunk = self._trunk(inputs)
        policy_logits = self.policy_linear(self._policy_features(trunk))
        wdl_logits = self._value_logits(trunk)
        state_value = HarbiChessActionValueNetwork.expected_wdl_value(wdl_logits)
        return policy_logits, wdl_logits, self.action_value_head(trunk, state_value)
