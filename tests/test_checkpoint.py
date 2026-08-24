from dataclasses import replace
from pathlib import Path

import pytest
from test_replay_schema import scripted_game

from harbichess.backends.mlx_network import HarbiChessNetwork, NetworkConfig
from harbichess.replay.schema import records_from_game
from harbichess.training.batch import GameBalancedSampler, build_training_batch
from harbichess.training.checkpoint import (
    CheckpointIntegrityError,
    load_training_checkpoint,
    save_training_checkpoint,
)
from harbichess.training.learner import LearnerConfig, MLXLearner
from harbichess.training.resume import ResumeState

mx = pytest.importorskip("mlx.core")


def _resume_state(*, step: int) -> ResumeState:
    return ResumeState(
        schema_version=1,
        run_id="pilot",
        checkpoint_id=f"step-{step}",
        source_commit="a" * 40,
        created_at="2026-08-24T12:00:00Z",
        training_step=step,
        lifetime_games=2,
        generation_games=2,
        training_elapsed_seconds=1.5,
        replay_samples=8,
        replay_cursor=7,
        model_file="model.safetensors",
        optimizer_file="optimizer.safetensors",
        rng_file="sampler-rng.json",
    )


def _learner() -> MLXLearner:
    network = HarbiChessNetwork(NetworkConfig(trunk_channels=8, residual_blocks=1))
    return MLXLearner(
        network,
        config=LearnerConfig(learning_rate=0.01, weight_decay=0.0),
    )


def test_checkpoint_restores_exact_learner_and_sampler_state(tmp_path: Path) -> None:
    _, game = scripted_game()
    first = records_from_game(game, run_id="pilot")
    second = tuple(replace(record, game_id="pilot-000000000002", game_index=2) for record in first)
    records = (*first, *second)
    sampler = GameBalancedSampler(records, seed=9)
    learner = _learner()
    learner.train_step(build_training_batch(sampler.sample(2)))
    expected_next = sampler.sample(3)
    sampler.set_rng_state(GameBalancedSampler(records, seed=9).rng_state)
    learner_for_save = _learner()
    learner_for_save.train_step(build_training_batch(sampler.sample(2)))

    checkpoint = tmp_path / "step-1"
    saved = save_training_checkpoint(
        checkpoint,
        state=_resume_state(step=1),
        learner=learner_for_save,
        sampler=sampler,
    )
    restored_learner = _learner()
    restored_sampler = GameBalancedSampler(records, seed=999)
    loaded = load_training_checkpoint(
        checkpoint,
        learner=restored_learner,
        sampler=restored_sampler,
    )

    assert loaded == saved
    assert restored_learner.step == 1
    assert restored_sampler.sample(3) == expected_next
    batch = build_training_batch(first[:1])
    assert restored_learner.evaluate_loss(batch) == pytest.approx(
        learner_for_save.evaluate_loss(batch)
    )
    uninterrupted_metrics = learner_for_save.train_step(batch)
    resumed_metrics = restored_learner.train_step(batch)
    assert resumed_metrics.total_loss == pytest.approx(uninterrupted_metrics.total_loss)
    assert restored_learner.evaluate_loss(batch) == pytest.approx(
        learner_for_save.evaluate_loss(batch)
    )
    assert sorted(path.name for path in checkpoint.iterdir()) == [
        "model.safetensors",
        "optimizer.safetensors",
        "resume.json",
        "sampler-rng.json",
    ]


def test_checkpoint_rejects_tampered_artifact(tmp_path: Path) -> None:
    _, game = scripted_game()
    records = records_from_game(game, run_id="pilot")
    learner = _learner()
    sampler = GameBalancedSampler(records, seed=3)
    checkpoint = tmp_path / "step-0"
    save_training_checkpoint(
        checkpoint,
        state=_resume_state(step=0),
        learner=learner,
        sampler=sampler,
    )
    (checkpoint / "sampler-rng.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="failed verification"):
        load_training_checkpoint(
            checkpoint,
            learner=_learner(),
            sampler=GameBalancedSampler(records, seed=4),
        )
