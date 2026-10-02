import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from harbichess.backends.torch_network import TorchChessNetwork
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.network_config import NetworkConfig
from harbichess.training.batch import TrainingBatch
from harbichess.training.config import LearnerConfig, NonFiniteTrainingError
from harbichess.training.torch_learner import TorchLearner


def batch():
    rules = PythonChessRules()
    position = BoardEncoder(rules).encode(rules.initial_state())
    policy = [0.0] * 4672
    policy[1], policy[3] = 0.75, 0.25
    mask = tuple(i in (1, 3, 5) for i in range(4672))
    return TrainingBatch(
        (position,) * 3, (tuple(policy),) * 3, (mask,) * 3, (0, 1, 2), (1.0, 0.0, 1.0)
    )


def learner():
    torch.set_num_threads(1)
    torch.manual_seed(51)
    return TorchLearner(
        TorchChessNetwork(
            NetworkConfig(
                trunk_channels=4,
                residual_blocks=1,
                policy_channels=2,
                value_channels=2,
                value_hidden=4,
            )
        ),
        config=LearnerConfig(learning_rate=0.003, weight_decay=0.0),
    )


def test_real_learning_matches_loss_semantics_and_decreases():
    model = learner()
    prepared = model.prepare_batch(batch())
    with torch.no_grad():
        policy, wdl = model.network(prepared.inputs)
    lp = torch.log_softmax(policy[:, [1, 3, 5]], 1).numpy()
    lv = torch.log_softmax(wdl, 1).numpy()
    expected_policy = -(0.75 * lp[:, 0] + 0.25 * lp[:, 1]).mean()
    expected_value = -(lv[0, 0] + lv[2, 2]) / 2
    before = model.evaluate_loss(prepared)
    np.testing.assert_allclose(
        before, (expected_policy + expected_value, expected_policy, expected_value), rtol=1e-6
    )
    initial = copy.deepcopy(model.network.state_dict())
    for _ in range(32):
        metrics = model.train_step(prepared)
    after = model.evaluate_loss(prepared)
    # Identical inputs have conflicting W/L labels; test a measurable update, not memorization.
    assert after[0] < before[0] - 0.05
    assert metrics.step == 32 and np.isfinite(metrics.gradient_norm)
    assert any(not torch.equal(v, model.network.state_dict()[k]) for k, v in initial.items())


def test_unknown_outcomes_have_no_value_gradient_and_nonfinite_update_is_rejected():
    model = learner()
    prepared = model.prepare_batch(replace(batch(), value_weights=(0.0,) * 3))
    old_value = {
        k: v.clone() for k, v in model.network.state_dict().items() if k.startswith("value_")
    }
    assert model.evaluate_loss(prepared)[2] == 0.0
    model.train_step(prepared)
    assert all(torch.equal(v, model.network.state_dict()[k]) for k, v in old_value.items())
    snapshot = copy.deepcopy(model.network.state_dict())
    optimizer = copy.deepcopy(model.optimizer.state_dict())
    prepared.inputs[0, 0, 0, 0] = float("nan")
    with pytest.raises(NonFiniteTrainingError):
        model.train_step(prepared)
    assert model.step == 1
    assert all(torch.equal(v, model.network.state_dict()[k]) for k, v in snapshot.items())
    assert optimizer["state"].keys() == model.optimizer.state_dict()["state"].keys()
    for key, values in optimizer["state"].items():
        for name, value in values.items():
            assert torch.equal(value, model.optimizer.state_dict()["state"][key][name])
