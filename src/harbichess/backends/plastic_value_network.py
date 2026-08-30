"""Stable MIHVER value base with a function-preserving plastic residual pathway."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.backends.invariant_value_network import InvariantValueConfig
from harbichess.backends.mlx_network import NetworkConfig, ResidualBlock
from harbichess.chess.encoding import METADATA_PLANES, PIECE_PLANES_PER_STEP


@dataclass(frozen=True, slots=True)
class PlasticValueConfig:
    """Capacity of the value-only pathway added after the qualified MIHVER base."""

    invariant_hidden: int = 32
    tower_channels: int = 16
    tower_blocks: int = 2
    hidden: int = 64

    def __post_init__(self) -> None:
        if min(
            self.invariant_hidden,
            self.tower_channels,
            self.tower_blocks,
            self.hidden,
        ) <= 0:
            raise ValueError("plastic value dimensions must be positive")


class HarbiChessPlasticValueNetwork(HarbiChessDecoupledValueNetwork):
    """Learn fresh value evidence without rewriting the stable MIHVER pathway.

    The plastic output projection is initialized to zero, so wrapping a MIHVER
    checkpoint preserves both policy and WDL logits exactly. The new branch sees
    explicit global/invariant features and an independent spatial tower; it does
    not depend on mutable policy or release-trunk representations.
    """

    def __init__(
        self,
        config: NetworkConfig | None = None,
        *,
        invariant_config: InvariantValueConfig | None = None,
        plastic_config: PlasticValueConfig | None = None,
    ) -> None:
        super().__init__(config, invariant_config=invariant_config)
        self.plastic_config = plastic_config or PlasticValueConfig()
        feature_count = PIECE_PLANES_PER_STEP + METADATA_PLANES
        self.plastic_invariant_hidden = nn.Linear(
            feature_count, self.plastic_config.invariant_hidden
        )
        self.plastic_tower_stem = nn.Conv2d(
            self.config.input_channels,
            self.plastic_config.tower_channels,
            kernel_size=3,
            padding=1,
        )
        self.plastic_tower_blocks = [
            ResidualBlock(self.plastic_config.tower_channels)
            for _ in range(self.plastic_config.tower_blocks)
        ]
        combined_features = (
            self.plastic_config.invariant_hidden + 2 * self.plastic_config.tower_channels
        )
        self.plastic_value_hidden = nn.Linear(combined_features, self.plastic_config.hidden)
        self.plastic_value_output = nn.Linear(self.plastic_config.hidden, 3)
        self._zero_plastic_output()

    def _zero_plastic_output(self) -> None:
        self.plastic_value_output.weight = mx.zeros_like(self.plastic_value_output.weight)
        self.plastic_value_output.bias = mx.zeros_like(self.plastic_value_output.bias)

    @classmethod
    def from_mihver(
        cls,
        base: HarbiChessDecoupledValueNetwork,
        *,
        plastic_config: PlasticValueConfig | None = None,
    ) -> HarbiChessPlasticValueNetwork:
        target = cls(
            base.config,
            invariant_config=base.invariant_config,
            plastic_config=plastic_config,
        )
        target.load_weights(list(tree_flatten(base.parameters())), strict=False)
        target._zero_plastic_output()
        mx.eval(target.parameters())
        return target

    def _plastic_value_residual(self, inputs: mx.array) -> mx.array:
        invariant = nn.relu(
            self.plastic_invariant_hidden(self.invariant_features(inputs))
        )
        tower = nn.relu(self.plastic_tower_stem(inputs))
        for block in self.plastic_tower_blocks:
            tower = block(tower)
        spatial = mx.concatenate(
            (mx.mean(tower, axis=(1, 2)), mx.max(tower, axis=(1, 2))), axis=1
        )
        hidden = nn.relu(
            self.plastic_value_hidden(mx.concatenate((invariant, spatial), axis=1))
        )
        return self.plastic_value_output(hidden)

    def _value_residual(self, inputs: mx.array) -> mx.array:
        return super()._value_residual(inputs) + self._plastic_value_residual(inputs)

    def _unfreeze_plastic_value(self) -> None:
        self.plastic_invariant_hidden.unfreeze()
        self.plastic_tower_stem.unfreeze()
        for block in self.plastic_tower_blocks:
            block.unfreeze()
        self.plastic_value_hidden.unfreeze()
        self.plastic_value_output.unfreeze()

    def freeze_to_plastic_value(self) -> None:
        """Train only the new residual while keeping all MIHVER parameters stable."""

        self.freeze()
        self._unfreeze_plastic_value()

    def freeze_to_stable_continuous_heads(self) -> None:
        """Retain DEVRIYE policy plasticity and isolate fresh value learning."""

        self.freeze_to_plastic_value()
        self.policy_conv.unfreeze()
        self.policy_linear.unfreeze()

    def freeze_to_low_lr_continuous_heads(self) -> None:
        """Expose stable value heads for an externally gradient-scaled ablation."""

        self.freeze_to_stable_continuous_heads()
        self.invariant_value_linear.unfreeze()
        self.global_value_hidden.unfreeze()
        self.global_value_output.unfreeze()

    def freeze_to_mutable_continuous_heads(self) -> None:
        """Expose stable and plastic heads at equal rate for the control arm."""

        self.freeze_to_low_lr_continuous_heads()


PLASTIC_VALUE_PREFIXES = (
    "plastic_invariant_hidden.",
    "plastic_tower_stem.",
    "plastic_tower_blocks.",
    "plastic_value_hidden.",
    "plastic_value_output.",
)

MIHVER_VALUE_PREFIXES = (
    "invariant_value_linear.",
    "global_value_hidden.",
    "global_value_output.",
)
