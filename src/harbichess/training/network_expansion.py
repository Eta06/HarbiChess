"""Function-preserving depth and policy-head expansion for controlled transfer tests."""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig


def expand_network_function_preserving(
    source: HarbiChessNetwork,
    target_config: NetworkConfig,
) -> HarbiChessNetwork:
    """Expand residual depth or policy channels without changing initial logits."""

    source_config = source.config
    unchanged = (
        "input_channels",
        "trunk_channels",
        "value_channels",
        "value_hidden",
        "policy_size",
    )
    if any(
        getattr(source_config, name) != getattr(target_config, name) for name in unchanged
    ):
        raise ValueError("function-preserving expansion requires unchanged trunk and value shapes")
    if target_config.residual_blocks < source_config.residual_blocks:
        raise ValueError("function-preserving expansion cannot remove residual blocks")
    if (
        target_config.policy_channels < source_config.policy_channels
        or target_config.policy_channels % source_config.policy_channels
    ):
        raise ValueError("target policy channels must be a multiple of source policy channels")

    target = HarbiChessNetwork(target_config)
    source_weights = dict(tree_flatten(source.parameters()))
    target_weights = dict(tree_flatten(target.parameters()))
    expanded: dict[str, mx.array] = {}

    for name, target_value in target_weights.items():
        source_value = source_weights.get(name)
        if source_value is not None and source_value.shape == target_value.shape:
            expanded[name] = source_value
        elif name.startswith("blocks."):
            block_index = int(name.split(".", 2)[1])
            if block_index < source_config.residual_blocks:
                raise ValueError(f"unsupported shape change for existing residual weight {name}")
            expanded[name] = mx.zeros_like(target_value)

    old_channels = source_config.policy_channels
    new_channels = target_config.policy_channels
    if new_channels != old_channels:
        channel_map = mx.array(
            tuple(index % old_channels for index in range(new_channels)),
            dtype=mx.int32,
        )
        expanded["policy_conv.weight"] = mx.take(
            source_weights["policy_conv.weight"], channel_map, axis=0
        )
        expanded["policy_conv.bias"] = mx.take(
            source_weights["policy_conv.bias"], channel_map, axis=0
        )
        feature_map = mx.array(
            tuple(
                square * old_channels + channel % old_channels
                for square in range(64)
                for channel in range(new_channels)
            ),
            dtype=mx.int32,
        )
        multiplicity = new_channels // old_channels
        expanded["policy_linear.weight"] = (
            mx.take(source_weights["policy_linear.weight"], feature_map, axis=1)
            / multiplicity
        )
        expanded["policy_linear.bias"] = source_weights["policy_linear.bias"]

    missing = set(target_weights) - set(expanded)
    if missing:
        raise ValueError(
            f"function-preserving expansion left weights unresolved: {sorted(missing)}"
        )
    target.load_weights(list(expanded.items()))
    mx.eval(target.parameters())
    return target
