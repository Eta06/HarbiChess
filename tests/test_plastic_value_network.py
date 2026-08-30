import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

mx = pytest.importorskip("mlx.core")

from harbichess.backends.decoupled_value_network import (  # noqa: E402
    HarbiChessDecoupledValueNetwork,
)
from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig  # noqa: E402
from harbichess.backends.plastic_value_network import (  # noqa: E402
    MIHVER_VALUE_PREFIXES,
    PLASTIC_VALUE_PREFIXES,
    HarbiChessPlasticValueNetwork,
    PlasticValueConfig,
)


def _networks():
    config = NetworkConfig(
        trunk_channels=8,
        residual_blocks=1,
        policy_channels=2,
        value_channels=2,
        value_hidden=8,
    )
    base = HarbiChessNetwork(config)
    mihver = HarbiChessDecoupledValueNetwork.from_base(base)
    plastic = HarbiChessPlasticValueNetwork.from_mihver(
        mihver,
        plastic_config=PlasticValueConfig(
            invariant_hidden=8,
            tower_channels=4,
            tower_blocks=1,
            hidden=8,
        ),
    )
    return mihver, plastic


def test_zero_initialized_plastic_path_is_function_preserving() -> None:
    mihver, plastic = _networks()
    inputs = mx.random.normal((3, 8, 8, mihver.config.input_channels))

    base_policy, base_value = mihver(inputs)
    policy, value = plastic(inputs)
    mx.eval(base_policy, base_value, policy, value)

    assert mx.array_equal(policy, base_policy).item()
    assert mx.array_equal(value, base_value).item()


def test_stable_mode_exposes_policy_and_only_plastic_value_parameters() -> None:
    _, plastic = _networks()
    plastic.freeze_to_stable_continuous_heads()
    names = {name for name, _ in tree_flatten(plastic.trainable_parameters())}

    assert any(name.startswith("policy_conv.") for name in names)
    assert any(name.startswith("policy_linear.") for name in names)
    assert all(
        name.startswith(("policy_conv.", "policy_linear.", *PLASTIC_VALUE_PREFIXES))
        for name in names
    )
    assert not any(name.startswith(MIHVER_VALUE_PREFIXES) for name in names)


def test_low_lr_mode_exposes_mihver_and_plastic_value_parameters() -> None:
    _, plastic = _networks()
    plastic.freeze_to_low_lr_continuous_heads()
    names = {name for name, _ in tree_flatten(plastic.trainable_parameters())}

    assert all(any(name.startswith(prefix) for name in names) for prefix in MIHVER_VALUE_PREFIXES)
    assert all(any(name.startswith(prefix) for name in names) for prefix in PLASTIC_VALUE_PREFIXES)


def test_plastic_training_can_change_value_without_changing_policy() -> None:
    _, plastic = _networks()
    plastic.freeze_to_plastic_value()
    inputs = mx.random.normal((4, 8, 8, plastic.config.input_channels))
    policy_before, value_before = plastic(inputs)
    loss_and_grad = nn.value_and_grad(
        plastic,
        lambda network, batch: mx.mean(network(batch)[1][:, 0]),
    )
    _, gradients = loss_and_grad(plastic, inputs)
    optimizer = optim.SGD(learning_rate=0.01)
    optimizer.update(plastic, gradients)
    policy_after, value_after = plastic(inputs)
    mx.eval(policy_before, value_before, policy_after, value_after)

    assert mx.array_equal(policy_after, policy_before).item()
    assert not mx.array_equal(value_after, value_before).item()
