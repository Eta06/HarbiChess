from pathlib import Path

import mlx.core as mx
import pytest

from harbichess.training.policy_projection import (
    PolicyProjectionConfig,
    _projection_reasons,
    _target_entropy,
)


def _config() -> PolicyProjectionConfig:
    return PolicyProjectionConfig(
        policy_target_result=Path("targets.json"),
        dataset_result=Path("dataset.json"),
        run_result=Path("run.json"),
        train_shard=Path("train.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_target_entropy_matches_soft_distribution() -> None:
    targets = mx.array(((0.5, 0.5), (1.0, 0.0)), dtype=mx.float32)
    assert _target_entropy(targets) == pytest.approx(0.5 * 0.69314718, abs=1e-6)


def test_projection_gate_uses_reducible_gap_and_safety() -> None:
    quality = {
        "mean_teacher_policy_spearman": 0.40,
        "verified_delta_95_interval": (0.001, 0.05),
        "harmful_ratio": 0.10,
        "mean_verified_regret": 0.10,
        "best_action_coverage_top_16": 0.80,
    }
    assert _projection_reasons(
        quality,
        gap_fraction=0.20,
        maximum_gradient_norm=1.0,
        config=_config(),
    ) == ()
    assert _projection_reasons(
        quality,
        gap_fraction=0.19,
        maximum_gradient_norm=1.0,
        config=_config(),
    ) == ("reducible policy KL gap closure is below 20%",)


def test_projection_config_freezes_ordered_scale_grid() -> None:
    assert _config().scales == tuple(index / 10 for index in range(1, 11))
    with pytest.raises(ValueError, match="configuration"):
        PolicyProjectionConfig(
            policy_target_result=Path("targets.json"),
            dataset_result=Path("dataset.json"),
            run_result=Path("run.json"),
            train_shard=Path("train.jsonl.gz"),
            output_dir=Path("output"),
            scales=(0.2, 0.1),
        )
