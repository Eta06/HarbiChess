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
