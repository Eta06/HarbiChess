from pathlib import Path

import pytest

from harbichess.evaluation.agreement_target import (
    AgreementTargetConfig,
    _gate,
    common_direction_target,
)


def _config() -> AgreementTargetConfig:
    return AgreementTargetConfig(
        consistency_result=Path("consistency.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_common_direction_target_moves_only_jointly_supported_mass() -> None:
    raw = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    first = {"a": 0.2, "b": 0.5, "c": 0.2, "d": 0.1}
    second = {"a": 0.3, "b": 0.4, "c": 0.1, "d": 0.2}

    target = common_direction_target(raw, first, second)

    assert target == pytest.approx({"a": 0.3, "b": 0.4, "c": 0.2, "d": 0.1})
    assert sum(target.values()) == pytest.approx(1.0)


def test_common_direction_target_preserves_raw_without_shared_direction() -> None:
    raw = {"a": 0.5, "b": 0.5}
    first = {"a": 0.8, "b": 0.2}
    second = {"a": 0.2, "b": 0.8}

    assert common_direction_target(raw, first, second) == raw


def test_agreement_gate_keeps_denge_thresholds() -> None:
    passing = {
        "qualified_ratio": 0.20,
        "harmful_row_ratio": 0.10,
        "expected_delta_vs_raw_95_interval": (0.001, 0.08),
        "qualified_expected_delta_95_interval": (0.02, 0.20),
        "mean_anchor_to_target_tv": 0.125,
        "mean_expected_delta_vs_800": -0.01,
    }
    assert _gate(passing, _config()) == {"passed": True, "reasons": []}

    failed = {**passing, "qualified_ratio": 0.19, "mean_anchor_to_target_tv": 0.126}
    assert _gate(failed, _config())["reasons"] == [
        "qualified target ratio is below 20%",
        "mean anchor-to-target TV exceeds 0.125",
    ]
