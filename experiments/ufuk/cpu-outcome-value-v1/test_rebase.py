"""Real original-e8 policy preservation and fast critic/full-forward parity."""

import os
from pathlib import Path

import chess
import numpy as np
import torch
from features import invariants
from train import rebase

from harbichess.backends.torch_network import load_weights
from harbichess.chess.actions import legal_action_indices
from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder


def test_actual_e8_rebase_preserves_policy_and_63_parameter_value_forward():
    torch.set_num_threads(1)
    path = Path(os.environ["HARBICHESS_TEST_E8"])
    model = load_weights(path).eval()
    encoder = TorchArrayBoardEncoder()
    board = chess.Board()
    encoded = encoder.encode_board(board)
    x = torch.tensor(np.array(encoded.values).reshape(1, 8, 8, 104))
    actions = torch.tensor([legal_action_indices(board)])
    with torch.inference_mode():
        old_policy, old_value = model.masked_policy_value(x, actions)
    rebase(model)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 63
    with torch.inference_mode():
        policy, value = model.masked_policy_value(x, actions)
        assert policy.numpy().tobytes() == old_policy.numpy().tobytes()
        assert not torch.equal(old_value, value) and torch.equal(value, torch.zeros_like(value))
        model.invariant_value_linear.weight.copy_(torch.arange(60).reshape(3, 20) / 100)
        model.invariant_value_linear.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        _, full_value = model.masked_policy_value(x, actions)
        fast_value = model.invariant_value_linear(torch.tensor(invariants(board)[None]))
        assert full_value.numpy().tobytes() == fast_value.numpy().tobytes()
