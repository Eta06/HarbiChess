import pytest

from harbichess.chess.actions import POLICY_SIZE
from harbichess.chess.encoding import ENCODER_CHANNELS, BoardEncoder
from harbichess.chess.rules import PythonChessRules

mx = pytest.importorskip("mlx.core")
network_module = pytest.importorskip("harbichess.backends.mlx_network")
HarbiChessNetwork = network_module.HarbiChessNetwork
NetworkConfig = network_module.NetworkConfig


def test_network_outputs_policy_and_wdl_logits() -> None:
    network = HarbiChessNetwork(NetworkConfig(trunk_channels=16, residual_blocks=1))
    inputs = mx.zeros((2, 8, 8, ENCODER_CHANNELS))
    policy, wdl = network(inputs)
    mx.eval(policy, wdl)

    assert policy.shape == (2, POLICY_SIZE)
    assert wdl.shape == (2, 3)
    assert network.parameter_count > 0


def test_network_rejects_incorrect_input_shape() -> None:
    network = HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1))
    with pytest.raises(ValueError, match="network input must have shape"):
        network(mx.zeros((1, 8, 8, ENCODER_CHANNELS - 1)))


def test_network_config_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="trunk_channels must be positive"):
        NetworkConfig(trunk_channels=0)


def test_encoded_chess_state_runs_through_network() -> None:
    rules = PythonChessRules()
    encoded = BoardEncoder(rules).encode(rules.initial_state())
    inputs = mx.array(encoded.values).reshape((1, *encoded.shape))
    network = HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1))

    policy, wdl = network(inputs)
    mx.eval(policy, wdl)
    assert policy.shape == (1, POLICY_SIZE)
    assert wdl.shape == (1, 3)


def test_masked_policy_head_matches_selected_full_logits() -> None:
    mx.random.seed(53)
    network = HarbiChessNetwork(NetworkConfig(trunk_channels=16, residual_blocks=2))
    inputs = mx.random.uniform(shape=(2, 8, 8, ENCODER_CHANNELS))
    actions = mx.array(((0, 17, 900), (5, 42, POLICY_SIZE - 1)), dtype=mx.int32)

    full_policy, full_wdl = network(inputs)
    masked_policy, masked_wdl = network.masked_policy_value(inputs, actions)
    expected = mx.take_along_axis(full_policy, actions, axis=1)
    mx.eval(expected, masked_policy, full_wdl, masked_wdl)

    assert float(mx.max(mx.abs(masked_policy - expected)).item()) < 1e-5
    assert float(mx.max(mx.abs(masked_wdl - full_wdl)).item()) == 0.0


def test_masked_policy_head_rejects_invalid_action_shape() -> None:
    network = HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1))
    inputs = mx.zeros((1, 8, 8, ENCODER_CHANNELS))

    with pytest.raises(ValueError, match="masked actions"):
        network.masked_policy_value(inputs, mx.zeros((1, 0), dtype=mx.int32))
