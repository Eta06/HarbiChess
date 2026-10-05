"""Actual e8 policy inheritance, explicit architecture transition and production parity."""

import os
from pathlib import Path

import chess
import numpy as np
import torch
from features import invariants
from train import critic, penalty, rebase, verify_mask

from harbichess.backends.torch_network import load_weights, save_weights
from harbichess.chess.actions import legal_action_indices
from harbichess.training.torch_array_encoder import TorchArrayBoardEncoder


def test_actual_policy_preservation_and_portable_full_value_parity(tmp_path):
    torch.set_num_threads(1)
    old = load_weights(Path(os.environ["HARBICHESS_TEST_E8"])).eval()
    torch.manual_seed(20262455)
    model = rebase(old).eval()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 26912
    assert all(
        torch.equal(p, old.state_dict()[n])
        for n, p in model.state_dict().items()
        if not n.startswith("value_sparse_head.")
    )
    encoder = TorchArrayBoardEncoder()
    board = chess.Board()
    with torch.no_grad():
        for ply in range(12):
            x = torch.tensor(np.asarray(encoder.encode_board(board).values).reshape(1, 8, 8, 104))
            actions = torch.tensor([legal_action_indices(board)])
            op, _ = old.masked_policy_value(x, actions)
            p, v = model.masked_policy_value(x, actions)
            assert op.numpy().tobytes() == p.numpy().tobytes()
            assert not torch.count_nonzero(v)
            features = torch.tensor(invariants(board)[None])
            board.push(list(board.legal_moves)[ply % board.legal_moves.count()])
        model.value_sparse_head.feature.weight[:3].normal_(std=0.001)
        model.value_sparse_head.metadata.weight[:3, 1:].normal_(std=0.001)
        verify_mask(model)
        _, full = model.masked_policy_value(x, actions)
        fast = critic(model, features)
        assert full.numpy().tobytes() == fast.numpy().tobytes()
        path = tmp_path / "model.safetensors"
        save_weights(path, model, provenance={"method": "test weights-only sparse transfer"})
        restored = load_weights(path).eval()
        rp, rv = restored.masked_policy_value(x, actions)
        assert rp.numpy().tobytes() == p.numpy().tobytes()
        assert rv.numpy().tobytes() == full.numpy().tobytes()


def test_effective_coordinate_mask_and_spatial_shrinkage():
    torch.set_num_threads(1)
    old = load_weights(Path(os.environ["HARBICHESS_TEST_E8"])).eval()
    model = rebase(old)
    optimizer = torch.optim.AdamW(
        model.value_sparse_head.parameters(), lr=0.00002, weight_decay=0, foreach=False
    )
    board = chess.Board()
    x = torch.tensor(invariants(board)[None])
    assert penalty(model) == 0
    original = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(critic(model, x), torch.tensor([0])) + penalty(
            model
        )
        loss.backward()
        model.value_sparse_head.metadata.weight.grad[:, 0].zero_()
        optimizer.step()
        verify_mask(model)
    assert all(torch.equal(p, original[n]) for n, p in model.named_parameters() if n in original)
    assert not torch.count_nonzero(model.value_sparse_head.feature.weight[3:])
    assert torch.count_nonzero(model.value_sparse_head.feature.weight[:3])
    assert all(int(s["step"]) == 4 for s in optimizer.state.values()) and len(optimizer.state) == 3
