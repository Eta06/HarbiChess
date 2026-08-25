from dataclasses import replace

import pytest
from test_replay_schema import scripted_game

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.replay.schema import records_from_game
from harbichess.training.batch import build_training_batch
from harbichess.training.learner import LearnerConfig, MLXLearner
from harbichess.training.pilot import PilotConfig, PilotStopReason, run_sanity_pilot

mx = pytest.importorskip("mlx.core")


def test_sanity_pilot_requires_loss_improvement_without_validation_regression() -> None:
    mx.random.seed(23)
    _, game = scripted_game()
    records = records_from_game(game, run_id="pilot")
    train = records[:2]
    validation = tuple(
        replace(record, game_id="validation-000000000008", game_index=8) for record in records[:2]
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
        on_step=lambda metric, _validation: observed_steps.append(metric),
    )

    assert report.passed, report.reasons
    assert report.final_train_loss < report.initial_train_loss
    assert report.final_validation_loss <= report.initial_validation_loss * 1.5
    assert report.maximum_gradient_norm > 0
    assert len(observed_steps) == 30
    assert report.sampler_rng_state is not None
    assert report.best_validation_step == report.steps
    assert report.best_validation_loss == pytest.approx(report.final_validation_loss)
    assert report.stop_reason is PilotStopReason.MAX_STEPS
    assert report.validation_evaluations == 3
    assert 1 <= len(report.validation_candidates) <= 4
    assert report.validation_candidates[-1].step == report.best_validation_step
    assert all(
        later.step > earlier.step
        for earlier, later in zip(
            report.validation_candidates,
            report.validation_candidates[1:],
            strict=False,
        )
    )


def test_sanity_pilot_rejects_game_level_leakage() -> None:
    _, game = scripted_game()
    records = records_from_game(game, run_id="pilot")
    learner = MLXLearner(HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)))

    with pytest.raises(ValueError, match="leak"):
        run_sanity_pilot(learner, records[:1], records[1:2], config=PilotConfig(steps=1))


def test_sanity_pilot_accepts_matching_prebuilt_evaluation_batches() -> None:
    _, game = scripted_game()
    records = records_from_game(game, run_id="pilot")
    train = records[:2]
    validation = tuple(
        replace(record, game_id="validation-000000000008", game_index=8) for record in records[2:]
    )
    learner = MLXLearner(HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)))

    report = run_sanity_pilot(
        learner,
        train,
        validation,
        config=PilotConfig(steps=1, batch_size=2),
        train_evaluation=build_training_batch(train),
        validation_evaluation=build_training_batch(validation),
    )

    assert report.steps == 1


def test_sanity_pilot_restores_best_validation_step_after_early_stop() -> None:
    mx.random.seed(31)
    _, game = scripted_game()
    records = records_from_game(game, run_id="pilot")
    train = records[:2]
    validation = tuple(
        replace(record, game_id="validation-000000000009", game_index=9) for record in records[2:]
    )
    learner = MLXLearner(
        HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1)),
        config=LearnerConfig(learning_rate=0.02, weight_decay=0.0),
    )

    report = run_sanity_pilot(
        learner,
        train,
        validation,
        config=PilotConfig(
            steps=20,
            batch_size=2,
            validation_interval_steps=1,
            early_stopping_patience=2,
            minimum_validation_delta=100.0,
        ),
    )

    assert report.stopped_early
    assert report.attempted_steps == 2
    assert report.best_validation_step == 0
    assert report.steps == 0
    assert learner.step == 0
    assert report.validation_candidates == ()
    assert report.stop_reason is PilotStopReason.EARLY_STOPPING
    assert report.stale_validation_evaluations == 2
    assert report.last_validation_step == 2
