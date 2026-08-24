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
    finite = MLXLearner._tree_is_finite(
        {"first": mx.array([1.0, 2.0]), "second": mx.array([3.0])}
    )
    non_finite = MLXLearner._tree_is_finite(
        {"first": mx.array([1.0]), "second": mx.array([float("nan")])}
    )
    mx.eval(finite, non_finite)

    assert bool(finite.item())
    assert not bool(non_finite.item())
