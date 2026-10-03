"""Real Inductor masked inference with preserved legal and perspective boundaries."""

import chess
import pytest
import torch

from harbichess.backends.torch_backend import TorchPolicyValueBackend
from harbichess.backends.torch_network import TorchChessNetwork
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.network_config import NetworkConfig


def test_real_compiled_pairwise_mixed_legal_batches_and_guards():
    torch.set_num_threads(1)
    torch.manual_seed(193)
    network = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    eager, fused = TorchPolicyValueBackend(network), TorchPolicyValueBackend(network, compiled=True)
    boards = [
        chess.Board(),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        chess.Board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1"),
        chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
    ]
    positions = [BoardEncoder().encode_board(board) for board in boards]
    actions = [legal_action_indices(board) for board in boards]
    for indices in [(0,), (1,), (2,), (3,), (0, 1, 2, 3)]:
        p, a = [positions[i] for i in indices], [actions[i] for i in indices]
        expected, actual = eager.evaluate_masked(p, a), fused.evaluate_masked(p, a)
        for left, right in zip(expected, actual, strict=True):
            for x, y in [
                (left.policy_logits, right.policy_logits),
                (left.wdl_logits, right.wdl_logits),
            ]:
                torch.testing.assert_close(torch.tensor(x), torch.tensor(y), atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(
                    torch.tensor(x).softmax(0), torch.tensor(y).softmax(0), atol=2e-5, rtol=2e-5
                )
    assert fused.evaluate_masked([], []) == []
    for bad in [(), (-1,), (4672,)]:
        with pytest.raises(ValueError):
            fused.evaluate_masked(positions[:1], [bad])
    assert fused.capabilities.supports_compilation
    assert not eager.capabilities.supports_compilation
    with pytest.raises(ValueError, match="CPU pairwise"):
        TorchPolicyValueBackend(TorchChessNetwork(), compiled=True)
