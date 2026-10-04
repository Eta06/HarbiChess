import subprocess
import sys

import chess
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten

from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork
from harbichess.backends.policy_context_transfer import transfer as add_context
from harbichess.backends.sparse_value_transfer import transfer
from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256
from harbichess.backends.ufuk_width_transfer import widen
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.network_config import NetworkConfig
from harbichess.training.config import LearnerConfig
from harbichess.training.torch_checkpoint import load_checkpoint, save_checkpoint
from harbichess.training.torch_learner import TorchLearner, TorchTrainingBatch


def test_sparse_real_backend_gradients_updates_frozen_policy_and_full_torch_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1012)
    old = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    source, path = tmp_path / "old.safetensors", tmp_path / "sparse.safetensors"
    save_weights(source, old)
    original_hash, rng = sha256(source), torch.get_rng_state().clone()
    receipt = transfer(source, path, channels=16, hidden=8)
    assert torch.equal(rng, torch.get_rng_state()) and sha256(source) == original_hash
    assert not receipt["full_training_resume"] and not receipt["initial_value_identity"]
    adapted = load_weights(path)
    assert "value_sparse" not in old.specification
    assert adapted.specification["value_sparse"] == {"schema": 1, "channels": 16, "hidden": 8}
    boards = [chess.Board(), chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")]
    x = torch.tensor([BoardEncoder().encode_board(b).values for b in boards]).reshape(2, 8, 8, 104)
    before = old(x)
    assert torch.equal(before[0], adapted(x)[0]) and not torch.equal(before[1], adapted(x)[1])
    masks, targets = torch.zeros(2, 4672, dtype=torch.bool), torch.zeros(2, 4672)
    for i, board in enumerate(boards):
        actions = legal_action_indices(board)
        masks[i, list(actions)] = True
        targets[i, actions[0]], targets[i, actions[1]] = 0.8, 0.2
        index = torch.tensor([actions])
        assert torch.equal(
            old.masked_policy_value(x[i : i + 1], index)[0],
            adapted.masked_policy_value(x[i : i + 1], index)[0],
        )
    batch = TorchTrainingBatch(x, targets, masks, torch.tensor([0, 1]), torch.tensor([1.0, 0.0]))
    for name, parameter in adapted.named_parameters():
        parameter.requires_grad_(name.startswith("value_sparse_head."))
    frozen = {name: v.clone() for name, v in old.state_dict().items()}
    learner = TorchLearner(adapted, config=LearnerConfig(learning_rate=1e-3))
    losses = learner._loss(batch)
    losses[0].backward()
    assert float(adapted.value_sparse_head.feature.weight.grad.abs().sum()) > 0
    mlx = HarbiChessPairwiseNetwork.from_portable(path)
    mlx.freeze()
    mlx.value_sparse_head.unfreeze()
    mx_inputs, mx_targets, mx_masks = (mx.array(v.numpy()) for v in (x, targets, masks))

    def loss(model):
        policy, value = model(mx_inputs)
        masked = mx.where(mx_masks, policy, -1e9)
        return (
            -(mx_targets * (masked - mx.logsumexp(masked, axis=1, keepdims=True))).sum(1).mean()
            + nn.losses.cross_entropy(value[:1], mx.array([0])).mean()
        )

    mx_loss, gradients = nn.value_and_grad(mlx, loss)(mlx)
    mx.eval(mx_loss, gradients)
    assert float(mx_loss) == pytest.approx(float(losses[0].detach()), abs=2e-5)
    converted = dict(tree_flatten(gradients))
    for name, parameter in adapted.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None and name not in converted
        else:
            np.testing.assert_allclose(
                parameter.grad.numpy(), np.array(converted[name]), atol=3e-5, rtol=3e-4
            )
    mx_before = mlx(mx_inputs)
    mx.eval(mx_before)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    for _ in range(2):
        optimizer.update(mlx, gradients)
        mx.eval(mlx.parameters(), optimizer.state)
        _, gradients = nn.value_and_grad(mlx, loss)(mlx)
        mx.eval(gradients)
    mx_after = mlx(mx_inputs)
    mx.eval(mx_after)
    np.testing.assert_array_equal(np.array(mx_before[0]), np.array(mx_after[0]))
    assert not np.array_equal(np.array(mx_before[1]), np.array(mx_after[1]))
    learner.train_step(batch)
    assert torch.equal(before[0], adapted(x)[0])
    assert all(torch.equal(v, adapted.state_dict()[name]) for name, v in frozen.items())
    save_checkpoint(
        tmp_path / "resume",
        learner=learner,
        sampler=None,
        replay_paths=(),
        run_state={},
        run_config={"test": "sparse-value-v1"},
        source_commit="a" * 40,
    )
    indices = tuple(torch.randint(batch.size, (batch.size,)).tolist())
    learner.train_step(batch.select(indices))
    expected_rng = torch.get_rng_state().clone()
    expected = {name: value.clone() for name, value in adapted.state_dict().items()}
    restored, _, _ = load_checkpoint(
        tmp_path / "resume", expected_run_config={"test": "sparse-value-v1"}
    )
    assert tuple(torch.randint(batch.size, (batch.size,)).tolist()) == indices
    assert restored.train_step(batch.select(indices)).step == 2
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert all(
        torch.equal(value, restored.network.state_dict()[name]) for name, value in expected.items()
    )
    for identifier, entry in learner.optimizer.state_dict()["state"].items():
        for key, value in entry.items():
            assert torch.equal(value, restored.optimizer.state_dict()["state"][identifier][key])
    exported = tmp_path / "mlx.safetensors"
    mlx.save_portable(exported)
    roundtrip = load_weights(exported)
    assert roundtrip.specification == adapted.specification
    for a, b in zip(roundtrip(x), mlx(mx_inputs), strict=True):
        np.testing.assert_allclose(a.detach().numpy(), np.array(b), atol=2e-5, rtol=2e-5)
    for operation in (transfer, add_context, widen):
        with pytest.raises(ValueError, match="plain versioned pairwise"):
            operation(path, tmp_path / f"unsupported-{operation.__module__}.safetensors")
    with pytest.raises(FileExistsError):
        transfer(source, path)
    assert sha256(source) == original_hash


def test_sparse_active_columns_and_full_ep_square_not_just_flag():
    model = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1),
        architecture="pairwise",
        value_sparse={"schema": 1, "channels": 8, "hidden": 4},
    )
    boards = [chess.Board(f"4k3/8/8/3pPpP1/8/8/8/4K3 w - {ep} 0 1") for ep in ("d6", "f6")]
    x = torch.tensor([BoardEncoder().encode_board(b).values for b in boards]).reshape(2, 8, 8, 104)
    vectors = torch.cat((x[:, :, :, :12].reshape(2, 768), x[:, :, :, 101].reshape(2, 64)), 1)
    layer = model.value_sparse_head.feature
    for vector in vectors:
        active = vector.nonzero().reshape(-1)
        torch.testing.assert_close(
            layer(vector), layer.weight[:, active].sum(1) + layer.bias, atol=2e-5, rtol=2e-5
        )
    assert torch.equal(x[0, :, :, 96:].mean((0, 1)), x[1, :, :, 96:].mean((0, 1)))
    with torch.no_grad():
        for parameter in model.value_sparse_head.parameters():
            parameter.zero_()
        model.value_sparse_head.feature.weight[0, 768 + chess.D6] = 0.5
        model.value_sparse_head.hidden.weight[0, 0] = 1
        model.value_sparse_head.output.weight[0, 0] = 1
    value = model(x)[1]
    assert value[0, 0] == 0.5 and value[1, 0] == 0


def test_sparse_portable_mlx_without_torch(tmp_path):
    path = tmp_path / "sparse.safetensors"
    save_weights(
        path,
        TorchChessNetwork(
            NetworkConfig(trunk_channels=4, residual_blocks=1),
            architecture="pairwise",
            value_sparse={"schema": 1, "channels": 8, "hidden": 4},
        ),
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mlx.core as mx; "
            "from harbichess.backends.pairwise_network import HarbiChessPairwiseNetwork; "
            "mx.set_default_device(mx.cpu); "
            "m=HarbiChessPairwiseNetwork.from_portable(sys.argv[1]); "
            "p,v=m(mx.zeros((1,8,8,104))); mx.eval(p,v); "
            "assert p.shape==(1,4672) and v.shape==(1,3); "
            "assert bool(mx.all(mx.isfinite(p))) and bool(mx.all(mx.isfinite(v))); "
            "assert 'torch' not in sys.modules",
            str(path),
        ],
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "spec",
    [
        {"schema": 2, "channels": 128, "hidden": 64},
        {"schema": 1, "channels": 0, "hidden": 64},
        {"schema": 1, "channels": True, "hidden": 64},
        {"schema": 1, "channels": 128, "hidden": -1},
        {"schema": 1, "channels": 128, "hidden": 0.5},
        {"schema": 1, "channels": 128, "hidden": 64, "unknown": 0},
    ],
)
def test_sparse_rejects_unsupported_specifications(spec):
    with pytest.raises(ValueError):
        TorchChessNetwork(architecture="pairwise", value_sparse=spec)
    with pytest.raises(ValueError):
        HarbiChessPairwiseNetwork(value_sparse=spec)
