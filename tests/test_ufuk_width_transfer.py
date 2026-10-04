import chess
import mlx.core as mx
import numpy as np
import pytest
import torch

from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork
from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256
from harbichess.backends.ufuk_width_transfer import widen
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.network_config import NetworkConfig


def test_real_widening_preserves_function_mlx_and_enables_new_features(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(611)
    mx.set_default_device(mx.cpu)
    old = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_channels=2, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    with torch.no_grad():
        for parameter in old.parameters():
            parameter.uniform_(-0.06, 0.06)
    source, destination = tmp_path / "old.safetensors", tmp_path / "wide.safetensors"
    save_weights(source, old)
    digest, rng = sha256(source), torch.get_rng_state().clone()
    receipt = widen(source, destination, trunk_channels=8, value_tower_channels=8)
    wide = load_weights(destination)
    assert torch.equal(rng, torch.get_rng_state()) and sha256(source) == digest
    assert receipt["optimizer"].startswith("reset")
    assert wide.config.trunk_channels == wide.invariant["channels"] == 8
    boards = [
        chess.Board(fen)
        for fen in (
            chess.STARTING_FEN,
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        )
    ]
    inputs = torch.tensor([BoardEncoder().encode_board(b).values for b in boards]).reshape(
        -1, 8, 8, 104
    )
    for a, b in zip(old(inputs), wide(inputs), strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    for i, board in enumerate(boards):
        actions = torch.tensor([legal_action_indices(board)])
        for a, b in zip(
            old.masked_policy_value(inputs[i : i + 1], actions),
            wide.masked_policy_value(inputs[i : i + 1], actions),
            strict=True,
        ):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    mlx = HarbiChessPairwiseNetwork.from_portable(destination)
    outputs = mlx(mx.array(inputs.numpy()))
    mx.eval(outputs)
    for a, b in zip(wide(inputs), outputs, strict=True):
        np.testing.assert_allclose(a.detach().numpy(), np.array(b), atol=2e-5, rtol=2e-5)
    optimizer = torch.optim.AdamW(wide.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        policy, value = wide(inputs)
        (
            policy[:, legal_action_indices(boards[0])].square().mean() + value.square().mean()
        ).backward()
        optimizer.step()
    assert float(wide.stem.weight.grad[4:].abs().sum()) > 0
    assert float(wide.value_tower_stem.weight.grad[4:].abs().sum()) > 0
    assert not torch.equal(old(inputs)[0], wide(inputs)[0])
    with pytest.raises(FileExistsError):
        widen(source, destination)
    with pytest.raises(ValueError, match="both widths"):
        widen(source, tmp_path / "invalid.safetensors", trunk_channels=4, value_tower_channels=8)
