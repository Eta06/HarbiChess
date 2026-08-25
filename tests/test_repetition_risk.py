import pytest

from harbichess.replay.repetition_risk import (
    risk_estimate,
    risk_gated_evidence,
    wilson_upper_bound,
)
from harbichess.replay.schema import BranchValueEstimate, ContinuationEvidence


def evidence() -> ContinuationEvidence:
    return ContinuationEvidence(
        method_version=1,
        confidence_level=0.95,
        branch_searches=8,
        simulations_per_search=64,
        repeat_value=0.0,
        minimum_advantage=0.01,
        repeat_actions=(1,),
        branches=(
            BranchValueEstimate(2, "a2a3", 8, 0.2, 0.02, 0.15, 0.25),
            BranchValueEstimate(3, "b2b3", 8, 0.2, 0.02, 0.15, 0.25),
        ),
        qualified_actions=(2, 3),
        source_model_sha256="a" * 64,
    )


def test_wilson_bound_requires_zero_events_to_clear_frozen_gate() -> None:
    zero = risk_estimate(
        action=2, horizon_plies=3, events=0, rollouts=16, confidence_level=0.95
    )
    one = risk_estimate(
        action=3, horizon_plies=3, events=1, rollouts=16, confidence_level=0.95
    )

    assert zero.upper_confidence_bound == pytest.approx(0.1446, abs=1e-4)
    assert one.upper_confidence_bound == pytest.approx(0.2375, abs=1e-4)
    assert wilson_upper_bound(0, 16, 0.95) < 0.20 < wilson_upper_bound(1, 16, 0.95)


def test_risk_gate_keeps_only_branches_whose_upper_bound_clears_threshold() -> None:
    risks = (
        risk_estimate(action=2, horizon_plies=3, events=0, rollouts=16, confidence_level=0.95),
        risk_estimate(action=3, horizon_plies=3, events=1, rollouts=16, confidence_level=0.95),
    )

    gated = risk_gated_evidence(evidence(), risks, maximum_repetition_risk=0.20)

    assert gated.method_version == 2
    assert gated.qualified_actions == (2,)
    assert gated.repetition_risks == risks


def test_risk_gate_requires_complete_previous_branch_coverage() -> None:
    risk = risk_estimate(
        action=2, horizon_plies=3, events=0, rollouts=16, confidence_level=0.95
    )

    with pytest.raises(ValueError, match="exactly cover"):
        risk_gated_evidence(evidence(), (risk,), maximum_repetition_risk=0.20)
