"""Required real framework parity tests; install the Linux parity extra, no skips."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from test_torch_learner import batch

from harbichess.backends.decoupled_value_network import HarbiChessDecoupledValueNetwork
from harbichess.backends.invariant_value_network import HarbiChessInvariantValueNetwork
from harbichess.backends.mlx_network import HarbiChessNetwork
from harbichess.backends.plastic_value_network import HarbiChessPlasticValueNetwork
from harbichess.backends.torch_network import TorchChessNetwork, save_weights
from harbichess.core.network_config import NetworkConfig
from harbichess.training.learner import MLXLearner
from harbichess.training.torch_learner import TorchLearner


@pytest.mark.parametrize(
    "architecture,mlx_class",
    [
        ("base", HarbiChessNetwork),
        ("invariant", HarbiChessInvariantValueNetwork),
        ("decoupled", HarbiChessDecoupledValueNetwork),
        ("plastic", HarbiChessPlasticValueNetwork),
    ],
)
def test_real_framework_forward_and_masked_policy_parity(tmp_path, architecture, mlx_class):
    torch.set_num_threads(1)
    torch.manual_seed(20261002)
    config = NetworkConfig(
        trunk_channels=4, residual_blocks=1, policy_channels=2, value_channels=2, value_hidden=4
    )
    t = TorchChessNetwork(config, architecture=architecture)
    with torch.no_grad():
        for p in t.parameters():
            p.uniform_(-0.03, 0.03)
    weights = tmp_path / "mlx.safetensors"
    save_weights(weights, t, mlx_layout=True)
    m = mlx_class(config)
    m.load_weights(str(weights))
    inputs = torch.rand(2, 8, 8, 104)
    actions = torch.tensor([[0, 123, 4671], [8, 50, 900]])
    with torch.no_grad():
        full = t(inputs)
        masked = t.masked_policy_value(inputs, actions)
    for method, expected in [
        (m, full),
        (lambda x: m.masked_policy_value(x, mx.array(actions.numpy())), masked),
    ]:
        actual = method(mx.array(inputs.numpy()))
        mx.eval(actual)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_allclose(np.array(a), e.numpy(), atol=2e-5, rtol=2e-5)


def test_real_framework_loss_and_gradient_parity(tmp_path):
    from mlx.utils import tree_flatten

    torch.set_num_threads(1)
    torch.manual_seed(32)
    config = NetworkConfig(
        trunk_channels=4, residual_blocks=1, policy_channels=2, value_channels=2, value_hidden=4
    )
    t = TorchChessNetwork(config)
    weights = tmp_path / "mlx.safetensors"
    save_weights(weights, t, mlx_layout=True)
    m = HarbiChessNetwork(config)
    m.load_weights(str(weights))
    tl, ml = TorchLearner(t), MLXLearner(m)
    data = batch()
    tp, mp = tl.prepare_batch(data), ml.prepare_batch(data)
    loss = tl._loss(tp)
    loss[0].backward()
    result, grad = nn.value_and_grad(m, ml._loss)(
        mp.inputs, mp.policy_targets, mp.legal_masks, mp.wdl_targets, mp.value_weights
    )
    mx.eval(result, grad)
    np.testing.assert_allclose(
        [float(x) for x in result], [float(x.detach()) for x in loss], atol=2e-5, rtol=2e-5
    )
    tg = dict(t.named_parameters())
    for name, value in tree_flatten(grad):
        array = np.array(value)
        if array.ndim == 4:
            array = array.transpose(0, 3, 1, 2)
        np.testing.assert_allclose(array, tg[name].grad.numpy(), atol=2e-5, rtol=1e-4)
