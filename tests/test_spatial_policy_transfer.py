from pathlib import Path

from harbichess.training.spatial_policy_transfer import (
    SpatialPolicyTransferConfig,
    _gate_reasons,
)


def _config() -> SpatialPolicyTransferConfig:
    return SpatialPolicyTransferConfig(
        policy_target_result=Path("targets.json"),
        dataset_result=Path("dataset.json"),
        run_result=Path("run.json"),
        train_shard=Path("train.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_spatial_policy_gate_requires_transfer_and_safety() -> None:
    quality = {
        "mean_teacher_policy_spearman": 0.40,
        "verified_delta_95_interval": (0.001, 0.05),
        "harmful_ratio": 0.10,
        "mean_verified_regret": 0.10,
        "best_action_coverage_top_16": 0.80,
    }
    assert _gate_reasons(
        quality,
        gap_fraction=0.20,
        maximum_gradient_norm=1.0,
        config=_config(),
    ) == ()

    failed = {**quality, "harmful_ratio": 0.101}
    assert _gate_reasons(
        failed,
        gap_fraction=0.20,
        maximum_gradient_norm=1.0,
        config=_config(),
    ) == ("harmful-action ratio exceeds 10%",)


def test_spatial_policy_fit_is_frozen_to_small_adapter_compute() -> None:
    config = _config()
    assert config.steps == 480
    assert config.batch_size == 16
    assert config.learning_rate == 1e-3
    assert config.seed == 2026082841
