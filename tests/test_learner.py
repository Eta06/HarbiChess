from dataclasses import replace

import pytest
from test_replay_schema import scripted_game

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.replay.schema import records_from_game
from harbichess.training.batch import build_training_batch
from harbichess.training.learner import LearnerConfig, MLXLearner

mx = pytest.importorskip("mlx.core")


def test_learner_overfits_one_replay_position_without_non_finite_values() -> None:
    mx.random.seed(17)
    _, game = scripted_game()
    record = records_from_game(game, run_id="pilot")[0]
    batch = build_training_batch((record,))
    network = HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1))
    learner = MLXLearner(
        network,
        config=LearnerConfig(learning_rate=0.01, weight_decay=0.0, max_gradient_norm=10.0),
    )
    initial = learner.evaluate_loss(batch)[0]

    metrics = [learner.train_step(batch) for _ in range(30)]
    final = learner.evaluate_loss(batch)[0]

    assert final < initial * 0.25
    assert all(metric.total_loss >= 0 for metric in metrics)
    assert all(metric.gradient_norm >= 0 for metric in metrics)
    assert learner.step == 30


def test_learner_configuration_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        LearnerConfig(learning_rate=float("nan"))
    with pytest.raises(ValueError, match="positive"):
        LearnerConfig(max_gradient_norm=0)


def test_gradient_finiteness_is_reduced_in_one_mlx_expression() -> None:
    finite = MLXLearner._tree_is_finite({"first": mx.array([1.0, 2.0]), "second": mx.array([3.0])})
    non_finite = MLXLearner._tree_is_finite(
        {"first": mx.array([1.0]), "second": mx.array([float("nan")])}
    )
    mx.eval(finite, non_finite)

    assert bool(finite.item())
    assert not bool(non_finite.item())


def test_learner_snapshot_restores_model_optimizer_and_step() -> None:
    mx.random.seed(29)
    _, game = scripted_game()
    batch = build_training_batch((records_from_game(game, run_id="pilot")[0],))
    learner = MLXLearner(HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)))
    learner.train_step(batch)
    snapshot = learner.snapshot()
    expected = learner.evaluate_loss(batch)

    learner.train_step(batch)
    learner.restore(snapshot)

    assert learner.step == 1
    assert learner.evaluate_loss(batch) == pytest.approx(expected)


def test_prepared_training_batch_selects_device_rows() -> None:
    _, game = scripted_game()
    batch = build_training_batch(records_from_game(game, run_id="prepared"))
    learner = MLXLearner(HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)))

    prepared = learner.prepare_batch(batch)
    selected = prepared.select((2, 0, 2))
    metrics = learner.train_step(selected)

    assert prepared.size == 4
    assert selected.size == 3
    assert metrics.step == 1
    with pytest.raises(IndexError, match="indices"):
        prepared.select(())


def test_unknown_outcomes_have_zero_value_loss_and_keep_policy_gradient() -> None:
    mx.random.seed(37)
    _, game = scripted_game()
    record = replace(records_from_game(game, run_id="truncated")[0], outcome_value=None)
    batch = build_training_batch((record,))
    learner = MLXLearner(HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)))

    total, policy, value = learner.evaluate_loss(batch)
    metric = learner.train_step(batch)

    assert value == pytest.approx(0.0)
    assert total == pytest.approx(policy)
    assert metric.policy_loss > 0.0
    assert metric.value_loss == pytest.approx(0.0)
