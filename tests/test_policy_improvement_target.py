from pathlib import Path

import pytest

from harbichess.evaluation.policy_improvement_target import (
    PolicyImprovementTargetConfig,
    _gate,
    mirror_descent_target,
)


def _config() -> PolicyImprovementTargetConfig:
    return PolicyImprovementTargetConfig(
        label_result=Path("labels.json"),
        dataset_result=Path("dataset.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_mirror_target_preserves_uncertain_mass_and_close_value_order() -> None:
    raw = {"best": 0.25, "close": 0.25, "worse": 0.25, "uncertain": 0.25}
    target, kl, _temperature = mirror_descent_target(
        raw,
        {"best": 0.55, "close": 0.54, "worse": 0.30},
        maximum_kl=0.10,
    )

    assert sum(target.values()) == pytest.approx(1.0)
    assert kl == pytest.approx(0.10, abs=1e-9)
    assert target["best"] > target["close"] > target["uncertain"] > target["worse"]
    assert target["best"] - target["close"] < target["close"] - target["worse"]
    assert target["uncertain"] > 0


def test_policy_improvement_gate_requires_verified_gain_without_collapse() -> None:
    passing = {
        "labelable_ratio": 0.99,
        "verified_expected_delta_95_interval": (0.001, 0.05),
        "maximum_target_to_raw_kl": 0.10,
        "harmful_expected_row_ratio": 0.05,
        "target_top_harmful_ratio": 0.05,
        "mean_target_top_verified_regret": 0.08,
        "mean_effective_action_ratio": 0.75,
    }
    assert _gate(passing, _config()) == {"passed": True, "reasons": []}

    failed = {**passing, "verified_expected_delta_95_interval": (-0.01, 0.05)}
    assert _gate(failed, _config())["reasons"] == [
        "verified expected-value improvement interval is not positive"
    ]


def test_policy_improvement_config_accepts_only_frozen_q_modes() -> None:
    assert _config().q_mode == "average"
    with pytest.raises(ValueError, match="configuration"):
        PolicyImprovementTargetConfig(
            label_result=Path("labels.json"),
            dataset_result=Path("dataset.json"),
            train_shard=Path("train.jsonl.gz"),
            validation_shard=Path("validation.jsonl.gz"),
            output_dir=Path("output"),
            q_mode="optimistic",  # type: ignore[arg-type]
        )


def test_policy_improvement_config_accepts_pairwise_teacher_evidence() -> None:
    config = PolicyImprovementTargetConfig(
        label_result=Path("labels.json"),
        dataset_result=Path("dataset.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
        pair_teacher_result=Path("pairs.json"),
        seed=2026082850,
    )
    assert config.pair_teacher_result == Path("pairs.json")
    assert config.seed == 2026082850
