import mlx.core as mx
import pytest

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.backends.spatial_policy_network import (
    HarbiChessSpatialPolicyNetwork,
    SpatialPolicyAdapter,
)
from harbichess.chess.actions import POLICY_SIZE
from harbichess.chess.encoding import ENCODER_CHANNELS


def test_spatial_policy_adapter_is_aligned_compact_and_function_preserving() -> None:
    mx.random.seed(97)
    config = NetworkConfig(trunk_channels=16, residual_blocks=2, policy_channels=4)
    base = HarbiChessNetwork(config)
    network = HarbiChessSpatialPolicyNetwork.from_base(base)
    inputs = mx.random.uniform(shape=(3, 8, 8, ENCODER_CHANNELS))

    base_policy, base_wdl = base(inputs)
    policy, wdl = network(inputs)
    trunk, frozen_policy, frozen_wdl = network.frozen_spatial_features(inputs)
    residual = network.spatial_policy_adapter(trunk)
    mx.eval(base_policy, base_wdl, policy, wdl, frozen_policy, frozen_wdl, residual)

    assert policy.shape == residual.shape == (3, POLICY_SIZE)
    assert float(mx.max(mx.abs(policy - base_policy)).item()) == 0.0
    assert float(mx.max(mx.abs(wdl - base_wdl)).item()) == 0.0
    assert float(mx.max(mx.abs(frozen_policy - base_policy)).item()) == 0.0
    assert float(mx.max(mx.abs(frozen_wdl - base_wdl)).item()) == 0.0
    assert network.parameter_count - base.parameter_count == 16 * 73 + 73


def test_spatial_policy_adapter_validates_shape_and_channels() -> None:
    with pytest.raises(ValueError, match="channels"):
        SpatialPolicyAdapter(0)
    adapter = SpatialPolicyAdapter(16)
    with pytest.raises(ValueError, match="requires"):
        adapter(mx.zeros((1, 64, 16)))
