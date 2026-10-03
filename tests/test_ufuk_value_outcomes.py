"""Real value updates isolate policy and match optimizer/RNG resume."""

import copy
from types import SimpleNamespace

import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork
from harbichess.core.network_config import NetworkConfig
from harbichess.training.config import LearnerConfig
from harbichess.training.torch_checkpoint import load_checkpoint, save_checkpoint
from harbichess.training.torch_learner import TorchLearner
from harbichess.training.ufuk_value_outcomes import (
    VALUE_PREFIXES,
    late_games,
    sample_indices,
    value_parameters,
    value_step,
)


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(406)
    network = TorchChessNetwork(
        NetworkConfig(trunk_channels=4, residual_blocks=1, value_hidden=4),
        architecture="pairwise",
        invariant={"channels": 4, "blocks": 1, "hidden": 4},
    )
    value_parameters(network)
    inputs = torch.randn(8, 8, 8, 104)
    panel = {"native_inputs": inputs, "own_groups": [[0, 1], [2, 3], [4, 5, 6, 7]]}
    targets = torch.tensor([[0.2, 0.6, 0.2]] * 8)
    return network, panel, inputs, targets


def test_only_value_updates_and_anchor_ignores_self_target():
    network, _, inputs, targets = setup()
    control = copy.deepcopy(network)
    candidate = copy.deepcopy(network)
    initial = {k: v.clone() for k, v in network.state_dict().items()}
    policy = network(inputs)[0].detach().clone()
    learner = TorchLearner(network, config=LearnerConfig(learning_rate=5e-5))
    other = TorchLearner(control, config=LearnerConfig(learning_rate=5e-5))
    own_learner = TorchLearner(candidate, config=LearnerConfig(learning_rate=5e-5))
    value_step(learner, inputs, targets, inputs, targets, include_self=False)
    value_step(other, inputs, targets, inputs, targets.roll(1, 1), include_self=False)
    value_step(own_learner, inputs, targets, inputs, targets.roll(1, 1), include_self=True)
    assert all(torch.equal(v, control.state_dict()[k]) for k, v in network.state_dict().items())
    assert all(
        torch.equal(v, network.state_dict()[k])
        for k, v in initial.items()
        if not k.startswith(VALUE_PREFIXES)
    )
    assert any(
        not torch.equal(v, network.state_dict()[k])
        for k, v in initial.items()
        if k.startswith(VALUE_PREFIXES)
    )
    assert torch.equal(policy, network(inputs)[0])
    assert torch.equal(policy, candidate(inputs)[0])
    assert any(
        not torch.equal(v, candidate.state_dict()[k])
        for k, v in network.state_dict().items()
        if k.startswith(VALUE_PREFIXES)
    )


def test_mixed_value_optimizer_and_sampling_resume_bitwise(tmp_path):
    initial, panel, inputs, targets = setup()
    direct = TorchLearner(copy.deepcopy(initial), config=LearnerConfig(learning_rate=5e-5))
    interrupted = TorchLearner(copy.deepcopy(initial), config=LearnerConfig(learning_rate=5e-5))
    rng = torch.get_rng_state().clone()

    def step(learner):
        native, own = sample_indices(panel, half_batch=4)
        value_step(
            learner,
            inputs[native],
            targets[native],
            inputs[own],
            targets[own].roll(1, 1),
            include_self=True,
        )
        return native, own

    direct_trace = [step(direct) for _ in range(4)]
    torch.set_rng_state(rng)
    split_trace = [step(interrupted) for _ in range(2)]
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        learner=interrupted,
        sampler=None,
        replay_paths=(),
        run_state={"cursor": 2},
        run_config={"experimental_loss": "native+self-value"},
        source_commit="0" * 40,
    )
    restored, _, _ = load_checkpoint(
        checkpoint, expected_run_config={"experimental_loss": "native+self-value"}
    )
    split_trace.extend(step(restored) for _ in range(2))
    assert split_trace == direct_trace
    assert restored.step == direct.step == 4
    assert all(
        torch.equal(v, restored.network.state_dict()[k])
        for k, v in direct.network.state_dict().items()
    )


def test_late_window_excludes_unknown_and_rejects_mixed_support():
    def rows(game, known, length):
        return [SimpleNamespace(game_id=game, ply=i, outcome_value=known) for i in range(length)]

    selected = late_games(rows("terminal", 1, 45) + rows("unknown", None, 60))
    assert [[r.ply for r in group] for group in selected] == [list(range(13, 45))]
    with pytest.raises(ValueError, match="mixed known/unknown"):
        late_games(rows("mixed", 1, 1) + rows("mixed", None, 2))
