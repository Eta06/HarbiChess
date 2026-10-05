"""Real CPU Torch/MLX additive-logit semantics and legacy schema safeguards."""

import chess
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from mlx.utils import tree_flatten

from harbichess.backends.numpy_sparse_value import NumpySparseValue
from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork
from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights
from harbichess.chess.actions import legal_action_indices
from harbichess.core.network_config import NetworkConfig, validate_sparse_value
from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder


def pair():
    torch.set_num_threads(1)
    torch.manual_seed(20262755)
    old = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    new = TorchChessNetwork.from_specification(
        {**old.specification, "value_sparse": {
            "schema": 2, "channels": 32, "hidden": 32, "composition": "additive-v1",
        }}
    )
    missing = new.load_state_dict(old.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert set(missing.missing_keys) == {
        k for k in new.state_dict() if k.startswith("value_sparse_head.")
    }
    new.requires_grad_(False)
    head = new.value_sparse_head
    head.feature.requires_grad_(True)
    head.metadata.requires_grad_(True)
    with torch.no_grad():
        head.feature.weight.zero_()
        head.feature.bias.zero_()
        head.feature.bias[:3] = 0.5
        head.metadata.weight.zero_()
        head.hidden.weight.copy_(torch.eye(32))
        head.hidden.bias.zero_()
        head.output.weight.zero_()
        head.output.weight[:3, :3].copy_(torch.eye(3) * 256)
        head.output.bias.fill_(-128)
    return old, new


def boards_and_inputs():
    white = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        white.push_uci(uci)
    black = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 b - - 37 1")
    ep = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    boards = [white, black, ep]
    inputs = torch.from_numpy(
        np.stack([TorchArrayBoardEncoder().encode_board(b).values for b in boards])
    ).reshape(len(boards), 8, 8, 104)
    return boards, inputs


def test_additive_zero_preserves_actual_base_and_masked_policy(tmp_path):
    old, new = pair()
    boards, x = boards_and_inputs()
    with torch.inference_mode():
        expected, actual = old(x), new(x)
        assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])
        for i, board in enumerate(boards):
            indices = torch.tensor([legal_action_indices(board)])
            a = old.masked_policy_value(x[i:i + 1], indices)
            b = new.masked_policy_value(x[i:i + 1], indices)
            assert all(torch.equal(u, v) for u, v in zip(a, b, strict=True))
    for layout in (False, True):
        path = tmp_path / f"residual-{layout}.safetensors"
        save_weights(path, new, mlx_layout=layout)
        restored = load_weights(path)
        assert restored.specification == new.specification
        assert all(torch.equal(a, b) for a, b in zip(restored(x), new(x), strict=True))


def test_additive_updates_only_residual_and_full_math_is_base_plus_head(tmp_path):
    old, new = pair()
    _, x = boards_and_inputs()
    frozen = {k: v.clone() for k, v in old.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [p for p in new.parameters() if p.requires_grad], lr=0.00002, weight_decay=0, foreach=False
    )
    before = new(x)[1].detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(new(x)[1], torch.tensor([0, 2, 1])).backward()
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in new.parameters())
        optimizer.step()
    assert not torch.equal(before, new(x)[1])
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in frozen.items())
    assert torch.equal(old(x)[0], new(x)[0])
    assert torch.equal(new(x)[1], old(x)[1] + new.value_sparse_head(x))
    path = tmp_path / "updated.safetensors"
    save_weights(path, new)
    restored = load_weights(path)
    assert all(torch.equal(a, b) for a, b in zip(restored(x), new(x), strict=True))
    with pytest.raises(ValueError, match="full inherited logits"):
        NumpySparseValue(path)


def test_real_mlx_cpu_additive_value_policy_and_gradient_parity(tmp_path):
    mx.set_default_device(mx.cpu)
    old, new = pair()
    boards, x = boards_and_inputs()
    with torch.no_grad():
        new.value_sparse_head.feature.weight[:3].uniform_(-0.0001, 0.0001)
    path = tmp_path / "additive.safetensors"
    save_weights(path, new)
    model = HarbiChessPairwiseNetwork.from_portable(path)
    model.freeze()
    model.value_sparse_head.feature.unfreeze()
    model.value_sparse_head.metadata.unfreeze()
    mx_x = mx.array(x.numpy())
    expected = new(x)
    actual = model(mx_x)
    mx.eval(actual)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.array(a), b.detach().numpy(), atol=3e-5, rtol=3e-5)
    actions = torch.tensor([legal_action_indices(boards[1])])
    a = model.masked_policy_value(mx_x[1:2], mx.array(actions.numpy()))
    b = new.masked_policy_value(x[1:2], actions)
    mx.eval(a)
    for u, v in zip(a, b, strict=True):
        np.testing.assert_allclose(np.array(u), v.detach().numpy(), atol=3e-5, rtol=3e-5)
    labels = torch.tensor([0, 2, 1])
    F.cross_entropy(expected[1], labels).backward()

    def loss(network):
        return nn.losses.cross_entropy(network(mx_x)[1], mx.array(labels.numpy())).mean()

    mlx_loss, gradients = nn.value_and_grad(model, loss)(model)
    mx.eval(mlx_loss, gradients)
    expected_loss = float(F.cross_entropy(expected[1], labels).detach())
    assert float(mlx_loss) == pytest.approx(expected_loss, abs=3e-5)
    converted = dict(tree_flatten(gradients))
    for name, param in new.named_parameters():
        if param.requires_grad:
            np.testing.assert_allclose(
                np.array(converted[name]), param.grad.numpy(), atol=0.002, rtol=0.0001
            )
        else:
            assert param.grad is None and name not in converted
    exported = tmp_path / "mlx-roundtrip.safetensors"
    model.save_portable(exported)
    restored = load_weights(exported)
    assert restored.specification == new.specification
    assert all(torch.equal(a, b) for a, b in zip(restored(x), expected, strict=True))
    assert torch.equal(old(x)[0], restored(x)[0])


def test_sparse_v1_preserved_and_ambiguous_v2_rejected():
    legacy = {"schema": 1, "channels": 32, "hidden": 32}
    assert validate_sparse_value(legacy, 104) == legacy
    good = {**legacy, "schema": 2, "composition": "additive-v1"}
    assert validate_sparse_value(good, 104) == good
    for bad in (
        {**legacy, "schema": 2},
        {**good, "composition": "replace"},
        {**good, "extra": 1},
        {**good, "schema": True},
        {**legacy, "composition": "additive-v1"},
    ):
        with pytest.raises(ValueError, match="sparse value"):
            validate_sparse_value(bad, 104)
