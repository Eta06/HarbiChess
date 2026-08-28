from pathlib import Path

import pytest

from harbichess.evaluation.uncertainty_q_labels import (
    UncertaintyQLabelConfig,
    _gate,
    uncertainty_labels,
)


def _config() -> UncertaintyQLabelConfig:
    return UncertaintyQLabelConfig(
        dataset_result=Path("dataset.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_uncertainty_labels_downweight_cross_budget_drift() -> None:
    labels = uncertainty_labels(
        {"a": 0.4, "b": 0.2, "c": 0.1},
        {"a": 0.4, "b": 0.215, "c": 0.14},
        {"a": 16, "b": 16, "c": 16},
        {"a": 16, "b": 16, "c": 16},
        drift_cutoff=0.03,
    )
    by_action = {action: (target, drift, weight) for action, target, drift, weight in labels}

    assert by_action["a"] == pytest.approx((0.4, 0.0, 2 / 3))
    assert by_action["b"] == pytest.approx((0.2075, 0.015, 1 / 3))
    assert by_action["c"][2] == 0.0
    assert sum(row[3] for row in labels) == pytest.approx(1.0)


def test_uncertainty_label_gate_requires_coverage_and_verified_strength() -> None:
    passing = {
        "labelable_ratio": 0.95,
        "mean_common_support_fraction": 0.95,
        "mean_stable_visit_mass": 0.80,
        "mean_stable_q_verified_spearman": 0.35,
        "conservative_verified_delta_95_interval": (0.001, 0.10),
        "conservative_harmful_ratio": 0.10,
        "mean_conservative_verified_regret": 0.10,
    }
    assert _gate(passing, _config()) == {"passed": True, "reasons": []}

    failed = {
        **passing,
        "mean_stable_visit_mass": 0.79,
        "conservative_verified_delta_95_interval": (-0.01, 0.10),
    }
    assert _gate(failed, _config())["reasons"] == [
        "drift-qualified visit mass is below 80%",
        "conservative-Q verified-improvement interval is not positive",
    ]


def test_uncertainty_label_gate_rejects_insufficient_labelable_coverage() -> None:
    summary = {
        "labelable_ratio": 0.94,
        "mean_common_support_fraction": 0.95,
        "mean_stable_visit_mass": 0.80,
        "mean_stable_q_verified_spearman": 0.35,
        "conservative_verified_delta_95_interval": (0.001, 0.10),
        "conservative_harmful_ratio": 0.10,
        "mean_conservative_verified_regret": 0.10,
    }
    assert _gate(summary, _config())["reasons"] == [
        "uncertainty-labelable position ratio is below 95%"
    ]
