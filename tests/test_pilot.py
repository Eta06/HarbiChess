from dataclasses import replace

import pytest
from test_replay_schema import scripted_game

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.replay.schema import records_from_game
from harbichess.training.learner import LearnerConfig, MLXLearner
from harbichess.training.pilot import PilotConfig, run_sanity_pilot

mx = pytest.importorskip("mlx.core")


def test_sanity_pilot_requires_loss_improvement_without_validation_regression() -> None:
    mx.random.seed(23)
    _, game = scripted_game()
    records = records_from_game(game, run_id="pilot")
    train = records[:2]
    validation = tuple(
        replace(record, game_id="validation-000000000008", game_index=8)
        for record in records[:2]
    )
    learner = MLXLearner(
        HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)),
        config=LearnerConfig(learning_rate=0.005, weight_decay=0.0),
    )
    observed_steps = []

    report = run_sanity_pilot(
        learner,
        train,
        validation,
        config=PilotConfig(
            steps=30,
            batch_size=2,
            minimum_train_improvement=0.1,
            maximum_validation_ratio=1.5,
            seed=7,
        ),
        on_step=observed_steps.append,
    )

    assert report.passed, report.reasons
    assert report.final_train_loss < report.initial_train_loss
    assert report.final_validation_loss <= report.initial_validation_loss * 1.5
    assert report.maximum_gradient_norm > 0
    assert len(observed_steps) == 30
    assert report.sampler_rng_state is not None


def test_sanity_pilot_rejects_game_level_leakage() -> None:
    _, game = scripted_game()
    records = records_from_game(game, run_id="pilot")
    learner = MLXLearner(
        HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1))
    )

    with pytest.raises(ValueError, match="leak"):
        run_sanity_pilot(learner, records[:1], records[1:2], config=PilotConfig(steps=1))
