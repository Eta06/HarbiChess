from pathlib import Path

import pytest

from harbichess.training.policy_convergence import PolicyConvergenceConfig


def _config() -> PolicyConvergenceConfig:
    return PolicyConvergenceConfig(
        policy_target_result=Path("targets.json"),
        dataset_result=Path("dataset.json"),
        run_result=Path("run.json"),
        train_shard=Path("train.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_policy_convergence_freezes_learning_curve() -> None:
    config = _config()
    assert config.checkpoint_steps == (480, 960, 1920)
    assert config.steps == 1920
    assert config.rank == 32
    assert config.learning_rate == 1e-3


def test_policy_convergence_rejects_unordered_checkpoints() -> None:
    with pytest.raises(ValueError, match="configuration"):
        PolicyConvergenceConfig(
            policy_target_result=Path("targets.json"),
            dataset_result=Path("dataset.json"),
            run_result=Path("run.json"),
            train_shard=Path("train.jsonl.gz"),
            output_dir=Path("output"),
            checkpoint_steps=(960, 480),
        )
