from harbichess.evaluation.consensus_target import (
    ConsensusTargetConfig,
    _expected_value,
    _gate,
    _jsd,
    _mixture,
    _normalize,
    _top_actions,
    _tv,
)


def _config() -> ConsensusTargetConfig:
    from pathlib import Path

    return ConsensusTargetConfig(
        consistency_result=Path("consistency.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_consensus_mixture_preserves_uncertain_support() -> None:
    import pytest

    first = {"a2a4": 0.7, "b2b4": 0.3}
    second = {"a2a4": 0.2, "b2b4": 0.5, "c2c4": 0.3}

    consensus = _mixture(first, second)

    assert consensus == pytest.approx({"a2a4": 0.45, "b2b4": 0.4, "c2c4": 0.15})
    assert _top_actions(consensus, 2) == ("a2a4", "b2b4")
    assert _tv(first, second) == 0.5
    assert _jsd(first, second) > 0
    assert _expected_value(
        consensus, {"a2a4": 0.8, "b2b4": 0.6, "c2c4": -0.2}
    ) == pytest.approx(0.57)


def test_consensus_policy_normalization_rejects_invalid_mass() -> None:
    import pytest

    assert _normalize({"a": 2.0, "b": 1.0}) == {"a": 2 / 3, "b": 1 / 3}
    with pytest.raises(ValueError, match="finite non-negative"):
        _normalize({"a": -1.0, "b": 2.0})


def test_consensus_gate_requires_stability_and_verified_value() -> None:
    passing = {
        "qualified_ratio": 0.25,
        "harmful_row_ratio": 0.05,
        "expected_delta_vs_raw_95_interval": (0.01, 0.08),
        "qualified_expected_delta_95_interval": (0.03, 0.12),
        "mean_anchor_to_consensus_tv": 0.10,
        "mean_expected_delta_vs_800": -0.005,
    }
    assert _gate(passing, _config()) == {"passed": True, "reasons": []}

    failed = {
        **passing,
        "qualified_ratio": 0.10,
        "expected_delta_vs_raw_95_interval": (-0.01, 0.08),
        "mean_expected_delta_vs_800": -0.02,
    }
    result = _gate(failed, _config())
    assert not result["passed"]
    assert result["reasons"] == [
        "qualified target ratio is below 20%",
        "full consensus expected-value interval is not positive",
        "consensus expected value regresses 800-search by more than 0.01",
    ]
