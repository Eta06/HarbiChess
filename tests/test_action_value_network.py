import mlx.core as mx
import pytest

from harbichess.backends.action_value_network import (
    ActionValueHead,
    HarbiChessActionValueNetwork,
)
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.actions import POLICY_SIZE
from harbichess.chess.encoding import ENCODER_CHANNELS


def test_action_value_expansion_preserves_policy_and_wdl_exactly() -> None:
    mx.random.seed(71)
    config = NetworkConfig(trunk_channels=16, residual_blocks=2, value_channels=2)
    base = HarbiChessNetwork(config)
    expanded = HarbiChessActionValueNetwork.from_base(base, action_value_channels=4)
    inputs = mx.random.uniform(shape=(3, 8, 8, ENCODER_CHANNELS))

    base_policy, base_wdl = base(inputs)
    policy, wdl, action_values = expanded.forward_with_action_values(inputs)
    expected_values = HarbiChessActionValueNetwork.expected_wdl_value(base_wdl)
    mx.eval(base_policy, base_wdl, policy, wdl, action_values, expected_values)

    assert float(mx.max(mx.abs(policy - base_policy)).item()) == 0.0
    assert float(mx.max(mx.abs(wdl - base_wdl)).item()) == 0.0
    assert action_values.shape == (3, POLICY_SIZE)
    assert float(mx.max(mx.abs(action_values - mx.tanh(expected_values[:, None]))).item()) == 0.0


def test_action_value_head_validates_shapes() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        ActionValueHead(16, 0, POLICY_SIZE)

    head = ActionValueHead(16, 4, POLICY_SIZE)
    with pytest.raises(ValueError, match="batched"):
        head(mx.zeros((2, 8, 8, 16)), mx.zeros((2, 1)))
