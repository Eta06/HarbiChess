"""Independent NHWC reference catches convolution/flatten/value conversion errors."""

import copy

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.backend import EncodedPosition
from harbichess.core.network_config import NetworkConfig


def nhwc_reference(inputs, weights, architecture, blocks, tower_blocks):
    def linear(x, name):
        return x @ weights[name + ".weight"].T + weights[name + ".bias"]

    def conv(x, name):
        w = weights[name + ".weight"]  # OHWI, independent of PyTorch's OIHW path.
        size = w.shape[1]
        padded = np.pad(x, ((0, 0), (size // 2, size // 2), (size // 2, size // 2), (0, 0)))
        result = np.zeros((*x.shape[:3], w.shape[0]), dtype=np.float32)
        for row in range(size):
            for col in range(size):
                result += np.einsum(
                    "bhwc,oc->bhwo",
                    padded[:, row : row + 8, col : col + 8],
                    w[:, row, col],
                    optimize=False,
                )
        return result + weights[name + ".bias"]

    def relu(x):
        return np.maximum(x, 0)

    def residual(x, name):
        return relu(x + conv(relu(conv(x, name + ".conv1")), name + ".conv2"))

    def tower(name, count):
        x = relu(conv(inputs, name + "_stem"))
        for i in range(count):
            x = residual(x, f"{name}_blocks.{i}")
        return np.concatenate((x.mean((1, 2)), x.max((1, 2))), 1)

    trunk = relu(conv(inputs, "stem"))
    for i in range(blocks):
        trunk = residual(trunk, f"blocks.{i}")
    policy = linear(relu(conv(trunk, "policy_conv")).reshape(len(inputs), -1), "policy_linear")
    value = linear(relu(conv(trunk, "value_conv")).reshape(len(inputs), -1), "value_hidden")
    value = linear(relu(value), "value_output")
    if architecture != "base":
        inv = np.concatenate((inputs[..., :12].sum((1, 2)), inputs[..., 96:].mean((1, 2))), 1)
        value += linear(inv, "invariant_value_linear")
        value += linear(
            relu(linear(tower("value_tower", tower_blocks), "value_tower_hidden")),
            "value_tower_output",
        )
    if architecture in ("decoupled", "plastic"):
        value += linear(relu(linear(inv, "global_value_hidden")), "global_value_output")
    if architecture == "plastic":
        hidden = np.concatenate(
            (relu(linear(inv, "plastic_invariant_hidden")), tower("plastic_tower", tower_blocks)), 1
        )
        value += linear(relu(linear(hidden, "plastic_value_hidden")), "plastic_value_output")
        value *= np.exp(weights["value_logit_scale"])
    return policy, value


@pytest.mark.parametrize("architecture", ["base", "invariant", "decoupled", "plastic"])
def test_mlx_weight_conversion_matches_independent_reference(tmp_path, architecture):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    network = TorchChessNetwork(
        NetworkConfig(
            trunk_channels=3, residual_blocks=1, policy_channels=2, value_channels=2, value_hidden=4
        ),
        architecture=architecture,
        invariant={"channels": 3, "blocks": 1, "hidden": 4},
        plastic={"channels": 3, "blocks": 1, "hidden": 4, "invariant_hidden": 3},
    )
    # Exercise every branch; zero-initialized residuals would conceal conversion errors.
    with torch.no_grad():
        for p in network.parameters():
            p.uniform_(-0.07, 0.07)
    weights = {
        k: (v.permute(0, 2, 3, 1) if v.ndim == 4 else v).contiguous()
        for k, v in network.state_dict().items()
    }
    legacy = tmp_path / "legacy.safetensors"
    save_file(weights, legacy)
    converted = load_weights(legacy, legacy_mlx=True)
    inputs = torch.rand(2, 8, 8, 104)
    expected = nhwc_reference(
        inputs.numpy(), {k: v.numpy() for k, v in weights.items()}, architecture, 1, 1
    )
    with torch.no_grad():
        actual = converted(inputs)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a.numpy(), e, atol=2e-5, rtol=2e-5)
    portable = tmp_path / "portable.safetensors"
    save_weights(portable, converted)
    exported = tmp_path / "exported.safetensors"
    save_weights(exported, load_weights(portable), mlx_layout=True)
    reloaded = load_weights(exported)
    for k, v in network.state_dict().items():
        assert torch.equal(v, reloaded.state_dict()[k]), k
    with pytest.raises(FileExistsError):
        save_weights(portable, converted)
    with pytest.raises(ValueError, match="unversioned"):
        load_weights(legacy)
    weights["unknown_head.weight"] = torch.ones(2, 2)
    save_file(weights, tmp_path / "unsupported.safetensors")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_weights(tmp_path / "unsupported.safetensors", legacy_mlx=True)


def test_backend_isolated_masked_forward_and_schema_validation():
    torch.set_num_threads(1)
    network = TorchChessNetwork(NetworkConfig(trunk_channels=4, residual_blocks=1))
    snapshot = copy.deepcopy(network.state_dict())
    backend = TorchPolicyValueBackend(network)
    assert network.training and next(network.parameters()).requires_grad
    position = BoardEncoder().encode(PythonChessRules().initial_state())
    full = backend.evaluate([position, position])
    actions = [(0, 44, 4671), (4,)]
    masked = backend.evaluate_masked([position, position], actions)
    for f, m, indices in zip(full, masked, actions, strict=True):
        np.testing.assert_allclose(
            m.policy_logits, [f.policy_logits[i] for i in indices], atol=2e-6
        )
        np.testing.assert_allclose(m.wdl_logits, f.wdl_logits, atol=2e-6)
    assert all(torch.equal(v, network.state_dict()[k]) for k, v in snapshot.items())
    assert backend.evaluate([]) == []
    with pytest.raises(ValueError, match="schema"):
        backend.evaluate([EncodedPosition(position.values, position.shape, 999)])
    with pytest.raises(ValueError, match="in range"):
        backend.evaluate_masked([position], [(4672,)])
    with pytest.raises(ValueError, match="shape"):
        network(torch.zeros(1, 104, 8, 8))
