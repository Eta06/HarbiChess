import mlx.core as mx
from mlx.utils import tree_flatten

from harbichess.backends.invariant_value_network import (
    HarbiChessInvariantValueNetwork,
    InvariantValueConfig,
)
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig


def _networks() -> tuple[HarbiChessNetwork, HarbiChessInvariantValueNetwork]:
    config = NetworkConfig(
        trunk_channels=8,
        residual_blocks=1,
        policy_channels=2,
        value_channels=2,
        value_hidden=8,
    )
    base = HarbiChessNetwork(config)
    target = HarbiChessInvariantValueNetwork.from_base(
        base,
        invariant_config=InvariantValueConfig(
            tower_channels=4,
            tower_blocks=1,
            tower_hidden=8,
        ),
    )
    return base, target


def test_invariant_expansion_preserves_full_policy_and_value_logits() -> None:
    mx.random.seed(7)
    base, target = _networks()
    inputs = mx.random.normal((3, 8, 8, base.config.input_channels))

    base_policy, base_value = base(inputs)
    target_policy, target_value = target(inputs)
    mx.eval(base_policy, base_value, target_policy, target_value)

    assert float(mx.max(mx.abs(base_policy - target_policy)).item()) == 0.0
    assert float(mx.max(mx.abs(base_value - target_value)).item()) == 0.0


def test_invariant_expansion_preserves_masked_evaluation() -> None:
    mx.random.seed(11)
    base, target = _networks()
    inputs = mx.random.normal((2, 8, 8, base.config.input_channels))
    actions = mx.array(((0, 17, 81), (5, 22, 99)), dtype=mx.int32)

    base_policy, base_value = base.masked_policy_value(inputs, actions)
    target_policy, target_value = target.masked_policy_value(inputs, actions)
    mx.eval(base_policy, base_value, target_policy, target_value)

    assert float(mx.max(mx.abs(base_policy - target_policy)).item()) == 0.0
    assert float(mx.max(mx.abs(base_value - target_value)).item()) == 0.0


def test_freeze_modes_never_expose_release_parameters() -> None:
    _, target = _networks()

    target.freeze_release_parameters()
    full_names = {name for name, _ in tree_flatten(target.trainable_parameters())}
    assert full_names
    assert all(
        name.startswith(
            (
                "invariant_value_linear.",
                "value_tower_stem.",
                "value_tower_blocks.",
                "value_tower_hidden.",
                "value_tower_output.",
            )
        )
        for name in full_names
    )

    target.freeze_to_global_linear()
    linear_names = {name for name, _ in tree_flatten(target.trainable_parameters())}
    assert linear_names == {
        "invariant_value_linear.weight",
        "invariant_value_linear.bias",
    }


def test_invariant_projection_receives_current_piece_counts() -> None:
    _, target = _networks()
    weights = [[0.0] * 20 for _ in range(3)]
    weights[0][0] = 1.0
    target.invariant_value_linear.weight = mx.array(weights)
    target.invariant_value_linear.bias = mx.zeros((3,))
    empty = mx.zeros((1, 8, 8, target.config.input_channels))
    one_piece = empty.at[0, 4, 5, 0].add(1.0)

    empty_value = target._value_residual(empty)
    piece_value = target._value_residual(one_piece)
    mx.eval(empty_value, piece_value)

    assert float((piece_value[0, 0] - empty_value[0, 0]).item()) == 1.0
    assert float((piece_value[0, 1] - empty_value[0, 1]).item()) == 0.0
