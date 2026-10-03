import chess
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten

from harbichess.backends.invariant_value_network import InvariantValueConfig
from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork
from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights
from harbichess.chess.actions import move_to_action
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.network_config import NetworkConfig
from harbichess.training.config import LearnerConfig
from harbichess.training.learner import MLXLearner, PreparedTrainingBatch
from harbichess.training.oracle_data import prepare_rows
from harbichess.training.torch_learner import TorchLearner


def test_real_mlx_cpu_pairwise_forward_masked_soft_loss_and_gradients(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(56)
    config = NetworkConfig(
        trunk_channels=4, residual_blocks=1, policy_channels=2, value_channels=2, value_hidden=4
    )
    network = TorchChessNetwork(
        config, architecture="pairwise", invariant={"channels": 4, "blocks": 1, "hidden": 4}
    )
    # Nonzero all branches; initialization zeros must not conceal conversion errors.
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.uniform_(-0.07, 0.07)
    mlx = HarbiChessPairwiseNetwork(config, invariant_config=InvariantValueConfig(4, 1, 4))
    exported = tmp_path / "pairwise-mlx.safetensors"
    save_weights(exported, network, mlx_layout=True)
    mlx.load_weights(str(exported), strict=True)
    board = chess.Board()
    board.push_uci("e2e4")
    legal = list(board.legal_moves)
    row = {
        "root_fen": chess.STARTING_FEN,
        "moves": ["e2e4"],
        "fen": board.fen(),
        "legal": [[m.uci(), move_to_action(board, m)] for m in legal],
        "policy": [
            [m.uci(), move_to_action(board, m), p]
            for m, p in zip(legal[:2], (0.7, 0.3), strict=True)
        ],
        "wdl": [0.2, 0.6, 0.2],
    }
    batch = prepare_rows([row, row])
    x = mx.array(batch.inputs.numpy())
    actual = network(batch.inputs)
    expected = mlx(x)
    mx.eval(expected)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a.detach().numpy(), np.array(e), atol=2e-5, rtol=2e-5)
    actions = torch.tensor([[move_to_action(board, m) for m in legal[:3]]] * 2)
    masked = network.masked_policy_value(batch.inputs, actions)
    masked_mx = mlx.masked_policy_value(x, mx.array(actions.numpy()))
    mx.eval(masked_mx)
    np.testing.assert_allclose(masked[0].detach().numpy(), np.array(masked_mx[0]), atol=2e-5)
    # Selected dot products reassociate FP32 sums; require numerical, not bitwise parity.
    torch.testing.assert_close(masked[0], actual[0].gather(1, actions), atol=2e-5, rtol=2e-5)
    learner = TorchLearner(network, config=LearnerConfig())
    losses = learner._loss(batch)
    losses[0].backward()
    prepared = PreparedTrainingBatch(
        *(
            mx.array(t.numpy())
            for t in (
                batch.inputs,
                batch.policies,
                batch.legal_masks,
                batch.wdl,
                batch.value_weights,
            )
        )
    )
    mlx_learner = MLXLearner(mlx)
    (mx_losses, gradients) = nn.value_and_grad(mlx, mlx_learner._loss)(
        *mlx_learner._prepared_arrays(prepared)
    )
    mx.eval(mx_losses, gradients)
    assert tuple(float(v) for v in mx_losses) == pytest.approx(
        tuple(float(v.detach()) for v in losses), abs=2e-5
    )
    converted_gradients = dict(tree_flatten(gradients))
    for key, parameter in network.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        if gradient.ndim == 4:
            gradient = gradient.permute(0, 2, 3, 1)
        np.testing.assert_allclose(
            gradient.numpy(), np.array(converted_gradients[key]), atol=3e-5, rtol=3e-4
        )
    restored = load_weights(exported)
    assert restored.architecture == "pairwise"
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in network.state_dict().items())


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 b - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ],
)
def test_pair_destination_preserves_special_move_action_geometry(fen):
    network = TorchChessNetwork(architecture="pairwise")
    board = chess.Board(fen)
    for move in board.legal_moves:
        action = move_to_action(board, move)
        assert network._geometric[action]
        target = move.to_square if board.turn else chess.square_mirror(move.to_square)
        assert network._destinations[action] == target
    assert BoardEncoder().encode_board(board).shape == (8, 8, 104)
