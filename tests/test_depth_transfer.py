import chess
import mlx.core as mx
import numpy as np
import pytest
import torch

from harbichess.backends.depth_transfer import deepen
from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork
from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.network_config import NetworkConfig


def test_identity_deepening_preserves_logits_and_enables_new_branch_learning(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(292)
    old = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_channels=2, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    with torch.no_grad():
        old.value_tower_output.weight.normal_(std=0.05)
    source, target = tmp_path / "old.safetensors", tmp_path / "deep.safetensors"
    save_weights(source, old)
    original_sha = sha256(source)
    rng = torch.get_rng_state().clone()
    receipt = deepen(source, target, extra_blocks=2)
    assert torch.equal(rng, torch.get_rng_state())
    deep = load_weights(target)
    assert deep.config.residual_blocks == deep.invariant["blocks"] == 3
    assert sha256(source) == original_sha
    assert receipt["optimizer"].startswith("reset")
    for k, v in old.state_dict().items():
        assert torch.equal(v, deep.state_dict()[k])
    boards = [
        chess.Board(fen)
        for fen in (
            chess.STARTING_FEN,
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        )
    ]
    x = torch.tensor(
        [BoardEncoder().encode_board(b).values for b in boards], dtype=torch.float32
    ).reshape(-1, 8, 8, 104)
    old_full, deep_full = old(x), deep(x)
    for a, b in zip(old_full, deep_full, strict=True):
        assert torch.equal(a, b)
    for i, board in enumerate(boards):
        actions = torch.tensor([legal_action_indices(board)])
        for a, b in zip(
            old.masked_policy_value(x[i : i + 1], actions),
            deep.masked_policy_value(x[i : i + 1], actions),
            strict=True,
        ):
            assert torch.equal(a, b)
    mlx = HarbiChessPairwiseNetwork.from_portable(target)
    outputs = mlx(mx.array(x.numpy()))
    mx.eval(outputs)
    for a, b in zip(deep_full, outputs, strict=True):
        np.testing.assert_allclose(a.detach().numpy(), np.array(b), atol=2e-5, rtol=2e-5)
    optimizer = torch.optim.AdamW(deep.parameters(), lr=1e-3)
    logits, value = deep(x)
    (logits[:, legal_action_indices(boards[0])].square().mean() + value.square().mean()).backward()
    for prefix in ("blocks.1", "value_tower_blocks.1"):
        gradient = dict(deep.named_parameters())[prefix + ".conv2.weight"].grad
        assert gradient is not None and float(gradient.abs().sum()) > 0
    optimizer.step()
    assert not torch.equal(deep(x)[1], deep_full[1])
    with pytest.raises(FileExistsError):
        deepen(source, target)
    with pytest.raises(ValueError):
        deepen(source, tmp_path / "bad.safetensors", extra_blocks=0)
