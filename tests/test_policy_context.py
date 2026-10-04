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
from harbichess.backends.policy_context_transfer import transfer
from harbichess.backends.torch_network import TorchChessNetwork, load_weights, save_weights, sha256
from harbichess.backends.ufuk_width_transfer import widen
from harbichess.chess.actions import legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.core.network_config import NetworkConfig
from harbichess.training.config import LearnerConfig
from harbichess.training.torch_checkpoint import load_checkpoint, save_checkpoint
from harbichess.training.torch_learner import TorchLearner, TorchTrainingBatch


def test_policy_context_real_backends_gradient_update_value_isolation_and_resume(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1012)
    old = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    source, path = tmp_path / "old.safetensors", tmp_path / "context.safetensors"
    save_weights(source, old)
    original_hash = sha256(source)
    rng = torch.get_rng_state().clone()
    receipt = transfer(source, path, heads=2)
    assert torch.equal(rng, torch.get_rng_state()) and sha256(source) == original_hash
    assert receipt["full_training_resume"] is False
    adapted = load_weights(path)
    assert "policy_context" not in old.specification
    assert adapted.specification["policy_context"] == {"schema": 1, "blocks": 2, "heads": 2}
    boards = [chess.Board(), chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")]
    x = torch.tensor([BoardEncoder().encode_board(b).values for b in boards]).reshape(2, 8, 8, 104)
    before = old(x)
    for a, b in zip(before, adapted(x), strict=True):
        assert torch.equal(a, b)
    for i, board in enumerate(boards):
        actions = torch.tensor([legal_action_indices(board)])
        for a, b in zip(
            old.masked_policy_value(x[i : i + 1], actions),
            adapted.masked_policy_value(x[i : i + 1], actions),
            strict=True,
        ):
            assert torch.equal(a, b)
    masks = torch.zeros(2, 4672, dtype=torch.bool)
    targets = torch.zeros(2, 4672)
    for i, board in enumerate(boards):
        actions = legal_action_indices(board)
        masks[i, list(actions)] = True
        targets[i, actions[0]] = 0.8
        targets[i, actions[1]] = 0.2
    batch = TorchTrainingBatch(x, targets, masks, torch.tensor([0, 1]), torch.tensor([1.0, 0.0]))
    for key, parameter in adapted.named_parameters():
        parameter.requires_grad_(key.startswith(("pair_", "policy_context_blocks.")))
    frozen = {k: v.clone() for k, v in adapted.named_parameters() if not v.requires_grad}
    learner = TorchLearner(adapted, config=LearnerConfig(learning_rate=1e-3))
    losses = learner._loss(batch)
    losses[0].backward()
    assert float(adapted.policy_context_blocks[0].proj.weight.grad.abs().sum()) > 0
    mlx = HarbiChessPairwiseNetwork.from_portable(path)
    mlx.freeze()
    for name in (
        "pair_hidden",
        "pair_query",
        "pair_key",
        "pair_origin_planes",
        "pair_destination_planes",
    ):
        getattr(mlx, name).unfreeze()
    for block in mlx.policy_context_blocks:
        block.unfreeze()
    mx_inputs, mx_targets, mx_masks = (mx.array(v.numpy()) for v in (x, targets, masks))

    def loss(model):
        policy, value = model(mx_inputs)
        policy = mx.where(mx_masks, policy, -1e9)
        return (
            -(mx_targets * (policy - mx.logsumexp(policy, axis=1, keepdims=True))).sum(1).mean()
            + nn.losses.cross_entropy(value[:1], mx.array([0])).mean()
        )

    mx_loss, gradients = nn.value_and_grad(mlx, loss)(mlx)
    mx.eval(mx_loss, gradients)
    assert float(mx_loss) == pytest.approx(float(losses[0].detach()), abs=2e-5)
    converted = dict(tree_flatten(gradients))
    for key, parameter in adapted.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None and key not in converted
            continue
        grad = parameter.grad
        if grad.ndim == 4:
            grad = grad.permute(0, 2, 3, 1)
        np.testing.assert_allclose(grad.numpy(), np.array(converted[key]), atol=3e-5, rtol=3e-4)
    mx_before = mlx(mx_inputs)[1]
    mx.eval(mx_before)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    optimizer.update(mlx, gradients)
    mx.eval(mlx.parameters(), optimizer.state)
    _, next_gradients = nn.value_and_grad(mlx, loss)(mlx)
    mx.eval(next_gradients)
    assert float(mx.abs(next_gradients["policy_context_blocks"][0]["qkv"]["weight"]).sum()) > 0
    optimizer.update(mlx, next_gradients)
    mx.eval(mlx.parameters(), optimizer.state)
    mx_after = mlx(mx_inputs)[1]
    mx.eval(mx_after)
    np.testing.assert_array_equal(np.array(mx_before), np.array(mx_after))
    learner.train_step(batch)
    assert torch.equal(before[1], adapted(x)[1]) and not torch.equal(before[0], adapted(x)[0])
    assert all(torch.equal(v, adapted.state_dict()[k]) for k, v in frozen.items())
    save_checkpoint(
        tmp_path / "resume",
        learner=learner,
        sampler=None,
        replay_paths=(),
        run_state={},
        run_config={"test": "context-v1"},
        source_commit="a" * 40,
    )
    sample_indices = tuple(torch.randint(batch.size, (batch.size,)).tolist())
    learner.train_step(batch.select(sample_indices))
    assert float(adapted.policy_context_blocks[0].qkv.weight.grad.abs().sum()) > 0
    expected_rng = torch.get_rng_state().clone()
    expected = {k: v.clone() for k, v in adapted.state_dict().items()}
    restored, _, _ = load_checkpoint(
        tmp_path / "resume", expected_run_config={"test": "context-v1"}
    )
    restored_indices = tuple(torch.randint(batch.size, (batch.size,)).tolist())
    assert restored_indices == sample_indices
    assert restored.train_step(batch.select(restored_indices)).step == 2
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, entry in learner.optimizer.state_dict()["state"].items():
        for key, value in entry.items():
            assert torch.equal(value, restored.optimizer.state_dict()["state"][name][key])
    assert all(torch.equal(v, restored.network.state_dict()[k]) for k, v in expected.items())
    exported = tmp_path / "mlx.safetensors"
    mlx.save_portable(exported)
    roundtrip = load_weights(exported)
    assert roundtrip.specification == adapted.specification
    for a, b in zip(roundtrip(x), mlx(mx_inputs), strict=True):
        np.testing.assert_allclose(a.detach().numpy(), np.array(b), atol=2e-5, rtol=2e-5)
    with pytest.raises(FileExistsError):
        transfer(source, path)
    with pytest.raises(ValueError):
        transfer(path, tmp_path / "double.safetensors")
    with pytest.raises(ValueError, match="plain versioned pairwise"):
        widen(path, tmp_path / "unsupported-width.safetensors")
    assert sha256(source) == original_hash


def test_policy_context_portable_mlx_load_and_forward_without_torch(tmp_path):
    path = tmp_path / "context.safetensors"
    model = TorchChessNetwork(
        architecture="pairwise", policy_context={"schema": 1, "blocks": 2, "heads": 4}
    )
    save_weights(path, model)
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
        {"schema": 2, "blocks": 2, "heads": 4},
        {"schema": 1, "blocks": 0, "heads": 4},
        {"schema": 1, "blocks": True, "heads": 4},
        {"schema": 1, "blocks": 2, "heads": 0},
        {"schema": 1, "blocks": 2, "heads": 3},
        {"schema": 1, "blocks": 2, "heads": True},
        {"schema": 1, "blocks": 2, "heads": 4, "unknown": 0},
    ],
)
def test_policy_context_rejects_unknown_or_invalid_specification(spec):
    with pytest.raises(ValueError):
        TorchChessNetwork(architecture="pairwise", policy_context=spec)
    with pytest.raises(ValueError):
        HarbiChessPairwiseNetwork(policy_context=spec)
