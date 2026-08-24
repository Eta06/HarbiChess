from pathlib import Path

import pytest

from harbichess.training.resume import ResumeState


def state() -> ResumeState:
    return ResumeState(
        schema_version=1,
        run_id="run-001",
        checkpoint_id="train-step-100",
        source_commit="a" * 40,
        created_at="2026-08-24T12:00:00Z",
        training_step=100,
        lifetime_games=2_000,
        generation_games=400,
        training_elapsed_seconds=3_600.5,
        replay_samples=120_000,
        replay_cursor=119_999,
        model_file="model.safetensors",
        optimizer_file="optimizer.safetensors",
        rng_file="rng.json",
    )


def test_resume_state_is_saved_atomically_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "resume.json"
    state().save_atomic(path)
    assert ResumeState.load(path) == state()
    assert list(tmp_path.iterdir()) == [path]


def test_resume_state_rejects_negative_counters() -> None:
    values = state().to_json().replace('"training_step": 100', '"training_step": -1')
    with pytest.raises(ValueError, match="cannot be negative"):
        ResumeState.from_json(values)

