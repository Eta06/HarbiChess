"""MLX implementation of the versioned compact origin/destination policy head."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.backends.invariant_value_network import InvariantValueConfig
from harbichess.chess.actions import action_destination_square
from harbichess.core.network_config import NetworkConfig


class HarbiChessPairwiseNetwork(HarbiChessDecoupledValueNetwork):
    """Same 104 input planes, 4672 policy actions and root-STM W/D/L logits."""

    def __init__(
        self,
        config: NetworkConfig | None = None,
        *,
        invariant_config: InvariantValueConfig | None = None,
    ) -> None:
        super().__init__(config, invariant_config=invariant_config)
        if self.config.policy_size != 4672:
            raise ValueError("pairwise policy requires canonical 4672-action schema")
        del self["policy_conv"], self["policy_linear"]
        self.pair_hidden = nn.Linear(self.config.trunk_channels + 34, 64)
        self.pair_query = nn.Linear(64, 32)
        self.pair_key = nn.Linear(64, 32)
        self.pair_origin_planes = nn.Linear(64, 73)
        self.pair_destination_planes = nn.Linear(64, 73)
        for layer in (self.pair_origin_planes, self.pair_destination_planes):
            layer.weight = mx.zeros_like(layer.weight)
            layer.bias = mx.zeros_like(layer.bias)
        destinations = [action_destination_square(i) for i in range(4672)]
        self._destinations = mx.array([s or 0 for s in destinations])
        self._geometric = mx.array([s is not None for s in destinations])
        self._coordinates = mx.array(
            [[2 * (s % 8) / 7 - 1, 2 * (s // 8) / 7 - 1] for s in range(64)]
        )

    def _pair_policy(self, inputs: mx.array, trunk: mx.array) -> mx.array:
        size = inputs.shape[0]
        squares = trunk.reshape(size, 64, -1)
        pieces = inputs[:, :, :, :12].reshape(size, 64, 12)
        global_features = self.invariant_features(inputs)
        normalized = mx.concatenate((global_features[:, :12] / 8, global_features[:, 12:]), axis=1)
        features = mx.concatenate(
            (
                squares,
                pieces,
                mx.broadcast_to(normalized[:, None, :], (size, 64, 20)),
                mx.broadcast_to(self._coordinates[None], (size, 64, 2)),
            ),
            axis=2,
        )
        hidden = nn.relu(self.pair_hidden(features))
        pair = (
            self.pair_query(hidden) @ mx.transpose(self.pair_key(hidden), (0, 2, 1)) / math.sqrt(32)
        )
        origins, planes = mx.arange(4672) // 73, mx.arange(4672) % 73
        logits = pair[:, origins, self._destinations]
        logits += self.pair_origin_planes(hidden)[:, origins, planes]
        logits += self.pair_destination_planes(hidden)[:, self._destinations, planes]
        return mx.where(self._geometric, logits, mx.array(-1e9))

    def __call__(self, inputs: mx.array) -> tuple[mx.array, mx.array]:
        trunk = self._trunk(inputs)
        return self._pair_policy(inputs, trunk), self._production_value_logits(inputs, trunk)

    def masked_policy_value(
        self, inputs: mx.array, action_indices: mx.array
    ) -> tuple[mx.array, mx.array]:
        if (
            action_indices.ndim != 2
            or action_indices.shape[0] != inputs.shape[0]
            or action_indices.shape[1] == 0
        ):
            raise ValueError("masked actions must have shape (batch, non-zero actions)")
        policy, value = self(inputs)
        return mx.take_along_axis(policy, action_indices, axis=1), value
