"""Function-preserving production WDL with a separate auxiliary material head."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from harbichess.backends.invariant_value_network import (
    HarbiChessInvariantValueNetwork,
    InvariantValueConfig,
)
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.encoding import HISTORY_STEPS, METADATA_PLANES, PIECE_PLANES_PER_STEP


class HarbiChessDecoupledValueNetwork(HarbiChessInvariantValueNetwork):
    """Separate deterministic representation supervision from production WDL logits."""

    def __init__(
        self,
        config: NetworkConfig | None = None,
        *,
        invariant_config: InvariantValueConfig | None = None,
    ) -> None:
        super().__init__(config, invariant_config=invariant_config)
        self.material_value_linear = nn.Linear(PIECE_PLANES_PER_STEP + METADATA_PLANES, 1)
        self.global_value_hidden = nn.Linear(PIECE_PLANES_PER_STEP + METADATA_PLANES, 64)
        self.global_value_output = nn.Linear(64, 3)
        self.material_value_linear.weight = mx.zeros_like(self.material_value_linear.weight)
        self.material_value_linear.bias = mx.zeros_like(self.material_value_linear.bias)
        self._zero_global_value_output()

    def _zero_global_value_output(self) -> None:
        self.global_value_output.weight = mx.zeros_like(self.global_value_output.weight)
        self.global_value_output.bias = mx.zeros_like(self.global_value_output.bias)

    @classmethod
    def from_base(
        cls,
        base: HarbiChessNetwork,
        *,
        invariant_config: InvariantValueConfig | None = None,
    ) -> HarbiChessDecoupledValueNetwork:
        target = cls(base.config, invariant_config=invariant_config)
        target.load_weights(list(tree_flatten(base.parameters())), strict=False)
        target._zero_residual_outputs()
        target.material_value_linear.weight = mx.zeros_like(target.material_value_linear.weight)
        target.material_value_linear.bias = mx.zeros_like(target.material_value_linear.bias)
        target._zero_global_value_output()
        mx.eval(target.parameters())
        return target

    @staticmethod
    def invariant_features(inputs: mx.array) -> mx.array:
        current_piece_counts = mx.sum(inputs[:, :, :, :PIECE_PLANES_PER_STEP], axis=(1, 2))
        metadata_start = HISTORY_STEPS * PIECE_PLANES_PER_STEP
        metadata = mx.mean(inputs[:, :, :, metadata_start:], axis=(1, 2))
        return mx.concatenate((current_piece_counts, metadata), axis=1)

    def material_value(self, inputs: mx.array) -> mx.array:
        return mx.tanh(self.material_value_linear(self.invariant_features(inputs))).squeeze(axis=1)

    def _value_residual(self, inputs: mx.array) -> mx.array:
        residual = super()._value_residual(inputs)
        hidden = nn.relu(self.global_value_hidden(self.invariant_features(inputs)))
        return residual + self.global_value_output(hidden)

    def freeze_to_material_head(self) -> None:
        self.freeze()
        self.material_value_linear.unfreeze()

    def freeze_to_global_wdl(self) -> None:
        self.freeze()
        self.invariant_value_linear.unfreeze()
        self.global_value_hidden.unfreeze()
        self.global_value_output.unfreeze()

    def freeze_to_global_tower_wdl(self) -> None:
        self.freeze_release_parameters()
        self.global_value_hidden.unfreeze()
        self.global_value_output.unfreeze()

    def freeze_to_continuous_heads(self) -> None:
        """Train policy and invariant WDL heads without disturbing their representations."""

        self.freeze()
        self.policy_conv.unfreeze()
        self.policy_linear.unfreeze()
        self.invariant_value_linear.unfreeze()
        self.global_value_hidden.unfreeze()
        self.global_value_output.unfreeze()
