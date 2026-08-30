import mlx.core as mx
from mlx.utils import tree_flatten

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig


def _networks() -> tuple[HarbiChessNetwork, HarbiChessDecoupledValueNetwork]:
    config = NetworkConfig(
        trunk_channels=8,
        residual_blocks=1,
        policy_channels=2,
        value_channels=2,
        value_hidden=8,
    )
    base = HarbiChessNetwork(config)
    return base, HarbiChessDecoupledValueNetwork.from_base(base)


def test_decoupled_expansion_preserves_production_logits() -> None:
    mx.random.seed(13)
    base, target = _networks()
    inputs = mx.random.normal((3, 8, 8, base.config.input_channels))

    base_policy, base_value = base(inputs)
    target_policy, target_value = target(inputs)
    material = target.material_value(inputs)
    mx.eval(base_policy, base_value, target_policy, target_value, material)

    assert float(mx.max(mx.abs(base_policy - target_policy)).item()) == 0.0
    assert float(mx.max(mx.abs(base_value - target_value)).item()) == 0.0
    assert material.tolist() == [0.0, 0.0, 0.0]


def test_material_training_cannot_change_production_wdl_parameters() -> None:
    _, target = _networks()

    target.freeze_to_material_head()
    names = {name for name, _ in tree_flatten(target.trainable_parameters())}

    assert names == {
        "material_value_linear.weight",
        "material_value_linear.bias",
    }


def test_wdl_training_cannot_change_material_head() -> None:
    _, target = _networks()

    target.freeze_to_global_tower_wdl()
    names = {name for name, _ in tree_flatten(target.trainable_parameters())}

    assert names
    assert not any(name.startswith("material_value_linear.") for name in names)
    assert {
        "global_value_hidden.weight",
        "global_value_hidden.bias",
        "global_value_output.weight",
        "global_value_output.bias",
    } <= names


def test_global_wdl_freeze_excludes_spatial_tower_and_release() -> None:
    _, target = _networks()

    target.freeze_to_global_wdl()
    names = {name for name, _ in tree_flatten(target.trainable_parameters())}

    assert names == {
        "invariant_value_linear.weight",
        "invariant_value_linear.bias",
        "global_value_hidden.weight",
        "global_value_hidden.bias",
        "global_value_output.weight",
        "global_value_output.bias",
    }


def test_continuous_freeze_trains_only_policy_and_invariant_wdl_heads() -> None:
    _, target = _networks()

    target.freeze_to_continuous_heads()
    names = {name for name, _ in tree_flatten(target.trainable_parameters())}

    assert names == {
        "policy_conv.weight",
        "policy_conv.bias",
        "policy_linear.weight",
        "policy_linear.bias",
        "invariant_value_linear.weight",
        "invariant_value_linear.bias",
        "global_value_hidden.weight",
        "global_value_hidden.bias",
        "global_value_output.weight",
        "global_value_output.bias",
    }
