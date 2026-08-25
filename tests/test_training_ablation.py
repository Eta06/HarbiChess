from pathlib import Path

import pytest

from harbichess.training.ablation import (
    AblationConfig,
    ContinuationTreatment,
    matched_filtered_fraction,
)


def test_filtered_fraction_preserves_per_record_exposure() -> None:
    fraction = matched_filtered_fraction(0.25, kept=15, total=378)

    assert fraction == pytest.approx(0.00992063492063492)
    assert fraction / 15 == pytest.approx(0.25 / 378)


def test_ablation_treatments_require_matching_shards() -> None:
    common = {
        "source_result": Path("source.json"),
        "train_shard": Path("train.gz"),
        "validation_shard": Path("validation.gz"),
    }

    with pytest.raises(ValueError, match="cannot receive"):
        AblationConfig(
            ablation_id="off",
            treatment=ContinuationTreatment.OFF,
            continuation_shards=(Path("continuation.gz"),),
            **common,
        )
    with pytest.raises(ValueError, match="requires at least one"):
        AblationConfig(
            ablation_id="current",
            treatment=ContinuationTreatment.CURRENT,
            **common,
        )
    with pytest.raises(ValueError, match="reference size"):
        AblationConfig(
            ablation_id="filtered",
            treatment=ContinuationTreatment.FILTERED,
            continuation_shards=(Path("continuation.gz"),),
            **common,
        )

    gated = AblationConfig(
        ablation_id="gated",
        treatment=ContinuationTreatment.CONFIDENCE_GATED,
        continuation_shards=(Path("confidence-v4.gz"),),
        **common,
    )
    assert gated.continuation_fraction == 0.25

    risk_gated = AblationConfig(
        ablation_id="risk-gated",
        treatment=ContinuationTreatment.REPETITION_RISK_GATED,
        continuation_shards=(Path("repetition-risk-v5.gz"),),
        **common,
    )
    assert risk_gated.continuation_fraction == 0.25

    value_aware = AblationConfig(
        ablation_id="value-aware",
        treatment=ContinuationTreatment.VALUE_AWARE_RISK,
        continuation_shards=(Path("value-aware-v6.gz"),),
        **common,
    )
    assert value_aware.continuation_fraction == 0.25

    value_regret = AblationConfig(
        ablation_id="value-regret",
        treatment=ContinuationTreatment.VALUE_REGRET,
        continuation_shards=(Path("value-regret-v7.gz"),),
        **common,
    )
    assert value_regret.continuation_fraction == 0.25
