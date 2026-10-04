"""MLX implementation of the versioned compact origin/destination policy head."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.backends.invariant_value_network import InvariantValueConfig
from harbichess.backends.mlx_context import PolicyContextBlock
from harbichess.backends.mlx_network import ResidualBlock
from harbichess.chess.actions import action_destination_square
from harbichess.core.network_config import (
    NetworkConfig,
    validate_policy_adapter,
    validate_policy_context,
)


class HarbiChessPairwiseNetwork(HarbiChessDecoupledValueNetwork):
    """Same 104 input planes, 4672 policy actions and root-STM W/D/L logits."""

    def __init__(
        self,
        config: NetworkConfig | None = None,
        *,
        invariant_config: InvariantValueConfig | None = None,
        policy_adapter: dict | None = None,
        policy_context: dict | None = None,
    ) -> None:
        super().__init__(config, invariant_config=invariant_config)
        self._policy_adapter = validate_policy_adapter(policy_adapter)
        self._policy_context = validate_policy_context(policy_context, self.config.trunk_channels)
        if self.config.policy_size != 4672:
            raise ValueError("pairwise policy requires canonical 4672-action schema")
        del self["policy_conv"], self["policy_linear"]
        if self._policy_context is not None:
            self.policy_context_blocks = [
                PolicyContextBlock(self.config.trunk_channels, self._policy_context["heads"])
                for _ in range(self._policy_context["blocks"])
            ]
        if self._policy_adapter is not None:
            self.policy_adapter_blocks = [
                ResidualBlock(self.config.trunk_channels)
                for _ in range(self._policy_adapter["blocks"])
            ]
            for block in self.policy_adapter_blocks:
                block.conv2.weight = mx.zeros_like(block.conv2.weight)
                block.conv2.bias = mx.zeros_like(block.conv2.bias)
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

    @classmethod
    def from_portable(cls, path: Path) -> HarbiChessPairwiseNetwork:
        """Load version-1 Torch OIHW or MLX OHWI weights without importing Torch.

        This restores model weights only, never optimizer/RNG/training state.
        Unversioned legacy models must use their original architecture loaders.
        """
        weights, header = mx.load(path, return_metadata=True)
        if "harbichess" not in header:
            raise ValueError("pairwise weights require versioned HarbiChess metadata")
        metadata = json.loads(header["harbichess"])
        if metadata["schema"] != 1:
            raise ValueError("unsupported weight schema")
        specification = metadata["specification"]
        if specification["architecture"] != "pairwise":
            raise ValueError("pairwise loader requires pairwise architecture")
        if metadata["layout"] == "torch-oihw":
            weights = {
                key: mx.transpose(value, (0, 2, 3, 1)) if value.ndim == 4 else value
                for key, value in weights.items()
            }
        elif metadata["layout"] != "mlx-ohwi":
            raise ValueError("unsupported tensor layout")
        if any(not bool(mx.all(mx.isfinite(value))) for value in weights.values()):
            raise ValueError("non-finite model weights")
        invariant = specification["invariant"]
        network = cls(
            NetworkConfig(**specification["config"]),
            invariant_config=InvariantValueConfig(
                invariant["channels"], invariant["blocks"], invariant["hidden"]
            ),
            policy_adapter=specification.get("policy_adapter"),
            policy_context=specification.get("policy_context"),
        )
        network.load_weights(list(weights.items()), strict=True)
        # Preserve even unused specification fields, without making them parameters.
        network._portable_specification = specification
        return network

    def save_portable(self, path: Path, *, provenance: dict | None = None) -> None:
        """Atomically publish versioned MLX weights for either backend, no overwrite."""
        if path.exists():
            raise FileExistsError(path)
        v = self.invariant_config
        specification = getattr(
            self,
            "_portable_specification",
            {
                "config": asdict(self.config),
                "architecture": "pairwise",
                "invariant": {
                    "channels": v.tower_channels,
                    "blocks": v.tower_blocks,
                    "hidden": v.tower_hidden,
                },
                "plastic": {"channels": 16, "blocks": 2, "hidden": 64, "invariant_hidden": 32},
            },
        )
        if self._policy_adapter is not None:
            specification = {**specification, "policy_adapter": dict(self._policy_adapter)}
        if self._policy_context is not None:
            specification = {**specification, "policy_context": dict(self._policy_context)}
        weights = dict(tree_flatten(self.parameters()))
        mx.eval(weights)
        if any(not bool(mx.all(mx.isfinite(value))) for value in weights.values()):
            raise ValueError("non-finite model weights")
        metadata = {
            "schema": 1,
            "layout": "mlx-ohwi",
            "specification": specification,
            "transfer": "weights-only",
            "provenance": provenance or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".safetensors")
        os.close(fd)
        try:
            mx.save_safetensors(temporary, weights, {"harbichess": json.dumps(metadata)})
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            os.unlink(temporary)

    def _pair_policy(
        self, inputs: mx.array, trunk: mx.array, actions: mx.array | None = None
    ) -> mx.array:
        for block in getattr(self, "policy_adapter_blocks", ()):
            trunk = block(trunk)
        size = inputs.shape[0]
        squares = trunk.reshape(size, 64, -1)
        for block in getattr(self, "policy_context_blocks", ()):
            squares = block(squares)
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
        if actions is not None:
            origins, destinations, planes = actions // 73, self._destinations[actions], actions % 73
            origin_hidden = mx.take_along_axis(hidden, origins[:, :, None], axis=1)
            destination_hidden = mx.take_along_axis(hidden, destinations[:, :, None], axis=1)
            logits = mx.sum(
                self.pair_query(origin_hidden) * self.pair_key(destination_hidden), axis=2
            ) / math.sqrt(32)
            for layer, vectors in (
                (self.pair_origin_planes, origin_hidden),
                (self.pair_destination_planes, destination_hidden),
            ):
                logits = (
                    logits + mx.sum(vectors * layer.weight[planes], axis=2) + layer.bias[planes]
                )
            return mx.where(self._geometric[actions], logits, mx.array(-1e9))
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
        trunk = self._trunk(inputs)
        return self._pair_policy(inputs, trunk, action_indices), self._production_value_logits(
            inputs, trunk
        )
