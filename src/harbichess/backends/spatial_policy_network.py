"""Function-preserving spatial policy-plane adapter for HarbiChess."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import POLICY_PLANES, POLICY_SIZE


class SpatialPolicyAdapter(nn.Module):
    """Shared residual policy logits aligned with origin-square move planes."""

    def __init__(self, trunk_channels: int) -> None:
        super().__init__()
        if trunk_channels <= 0:
            raise ValueError("spatial policy adapter channels must be positive")
        self.projection = nn.Conv2d(trunk_channels, POLICY_PLANES, kernel_size=1)
        self.projection.weight = mx.zeros_like(self.projection.weight)
        self.projection.bias = mx.zeros_like(self.projection.bias)

    def __call__(self, trunk: mx.array) -> mx.array:
        if trunk.ndim != 4 or tuple(trunk.shape[1:3]) != (8, 8):
            raise ValueError("spatial policy adapter requires (batch, 8, 8, channels)")
        logits = self.projection(trunk).reshape(trunk.shape[0], POLICY_SIZE)
        return logits


class HarbiChessSpatialPolicyNetwork(HarbiChessNetwork):
    """Release network plus a compact residual spatial policy head."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__(config)
        self.spatial_policy_adapter = SpatialPolicyAdapter(self.config.trunk_channels)

    @classmethod
    def from_base(cls, base: HarbiChessNetwork) -> HarbiChessSpatialPolicyNetwork:
        target = cls(base.config)
        target.load_weights(list(tree_flatten(base.parameters())), strict=False)
        mx.eval(target.parameters())
        return target

    def frozen_spatial_features(
        self, inputs: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        trunk = self._trunk(inputs)
        policy = self.policy_linear(self._policy_features(trunk))
        wdl = self._value_logits(trunk)
        return (
            mx.stop_gradient(trunk),
            mx.stop_gradient(policy),
            mx.stop_gradient(wdl),
        )

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        trunk = self._trunk(inputs)
        policy = self.policy_linear(self._policy_features(trunk))
        return policy + self.spatial_policy_adapter(trunk), self._value_logits(trunk)
