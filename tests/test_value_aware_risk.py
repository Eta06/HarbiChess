import pytest

from harbichess.replay.schema import BranchValueEstimate, ContinuationEvidence
from harbichess.replay.value_aware_risk import (
    loop_value_lower_bound,
    value_aware_evidence,
    value_aware_risk_estimate,
)


def branch(action: int = 2, lower: float = 0.15) -> BranchValueEstimate:
    return BranchValueEstimate(action, "a2a3", 8, 0.20, 0.02, lower, 0.25)


def evidence() -> ContinuationEvidence:
    return ContinuationEvidence(
        method_version=1,
        confidence_level=0.95,
        branch_searches=8,
        simulations_per_search=64,
        repeat_value=0.0,
        minimum_advantage=0.01,
        repeat_actions=(1,),
        branches=(branch(),),
        qualified_actions=(2,),
        source_model_sha256="a" * 64,
    )


def test_sparse_loop_value_keeps_uncertainty_explicit() -> None:
    assert loop_value_lower_bound((), 0.95) == (None, None)
    assert loop_value_lower_bound((0.4,), 0.95) == (0.4, -1.0)


def test_value_aware_risk_combines_probability_and_loop_value() -> None:
    safe = value_aware_risk_estimate(
        branch=branch(),
        horizon_plies=3,
        rollouts=16,
        loop_values=(),
        confidence_level=0.95,
        repeat_value=0.0,
    )
    costly = value_aware_risk_estimate(
        branch=branch(),
        horizon_plies=3,
        rollouts=16,
        loop_values=(-0.5,),
        confidence_level=0.95,
        repeat_value=0.0,
    )

    assert safe.risk_adjusted_value_lower_bound == pytest.approx(0.15)
    assert costly.risk_adjusted_value_lower_bound == pytest.approx(0.078125)
    assert costly.mean_loop_value == -0.5
    assert costly.lower_loop_value_bound == -1.0


def test_value_aware_gate_requires_advantage_and_expected_value() -> None:
    risk = value_aware_risk_estimate(
        branch=branch(),
        horizon_plies=3,
        rollouts=16,
        loop_values=(-0.5,),
        confidence_level=0.95,
        repeat_value=0.0,
    )

    gated = value_aware_evidence(
        evidence(), (risk,), root_value=0.08, minimum_advantaged_root_value=0.05
    )

    assert gated.method_version == 3
    assert gated.qualified_actions == (2,)
    assert gated.maximum_repetition_risk is None
    with pytest.raises(ValueError, match="defensive or equal"):
        value_aware_evidence(
            evidence(), (risk,), root_value=0.05, minimum_advantaged_root_value=0.05
        )
