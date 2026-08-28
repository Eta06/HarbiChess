import mlx.core as mx
import pytest

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.chess.encoding import ENCODER_CHANNELS
from harbichess.training.network_expansion import expand_network_function_preserving


@pytest.mark.parametrize(
    ("blocks", "policy_channels"),
    ((4, 4), (2, 8), (4, 8)),
)
def test_depth_and_policy_expansion_preserve_network_logits(
    blocks: int,
    policy_channels: int,
) -> None:
    mx.random.seed(41)
    source = HarbiChessNetwork(
        NetworkConfig(
            trunk_channels=16,
            residual_blocks=2,
            policy_channels=4,
            value_channels=2,
            value_hidden=32,
        )
    )
    target = expand_network_function_preserving(
        source,
        NetworkConfig(
            trunk_channels=16,
            residual_blocks=blocks,
            policy_channels=policy_channels,
            value_channels=2,
            value_hidden=32,
        ),
    )
    inputs = mx.random.uniform(shape=(3, 8, 8, ENCODER_CHANNELS))

    source_policy, source_value = source(inputs)
    target_policy, target_value = target(inputs)
    mx.eval(source_policy, source_value, target_policy, target_value)

    assert float(mx.max(mx.abs(source_policy - target_policy)).item()) < 1e-5
    assert float(mx.max(mx.abs(source_value - target_value)).item()) < 1e-5
    assert target.parameter_count >= source.parameter_count


def test_function_preserving_expansion_rejects_incompatible_shapes() -> None:
    source = HarbiChessNetwork(
        NetworkConfig(trunk_channels=16, residual_blocks=2, policy_channels=4)
    )

    with pytest.raises(ValueError, match="unchanged trunk"):
        expand_network_function_preserving(
            source,
            NetworkConfig(trunk_channels=32, residual_blocks=2, policy_channels=4),
        )
    with pytest.raises(ValueError, match="cannot remove"):
        expand_network_function_preserving(
            source,
            NetworkConfig(trunk_channels=16, residual_blocks=1, policy_channels=4),
        )
    with pytest.raises(ValueError, match="multiple"):
        expand_network_function_preserving(
            source,
            NetworkConfig(trunk_channels=16, residual_blocks=2, policy_channels=6),
        )
