import math
from dataclasses import replace

import pytest
from test_replay_schema import scripted_game

from harbichess.replay.schema import (
    BranchValueEstimate,
    ContinuationEvidence,
    RepetitionRiskEstimate,
    records_from_game,
)
from harbichess.replay.value_regret import blend_policy_by_regret


def inputs(root_value: float):
    rules, game = scripted_game()
    original = records_from_game(game, run_id="regret", rules=rules)[0]
    original = replace(
        original,
        policy=((original.selected_action, 0.5), (1, 0.5)),
    )
    branch = BranchValueEstimate(
        original.selected_action, "f2f3", 8, 0.2, 0.02, 0.15, 0.25
    )
    evidence = ContinuationEvidence(
        method_version=1,
        confidence_level=0.95,
        branch_searches=8,
        simulations_per_search=64,
        repeat_value=0.0,
        minimum_advantage=0.01,
        repeat_actions=(1,),
        branches=(branch,),
        qualified_actions=(original.selected_action,),
        source_model_sha256="a" * 64,
    )
    evidenced = replace(
        original,
        root_value=root_value,
        policy=((original.selected_action, 1.0),),
        continuation_evidence=evidence,
    )
    risk = RepetitionRiskEstimate(
        action=original.selected_action,
        horizon_plies=3,
        rollouts=16,
        repetition_events=1,
        estimated_risk=1 / 16,
        upper_confidence_bound=0.2375,
        loop_value_samples=1,
        exact_loop_value_samples=1,
        mean_loop_value=0.0,
        lower_loop_value_bound=0.0,
        risk_adjusted_value_lower_bound=0.14,
    )
    return original, evidenced, (risk,)


def test_losing_root_preserves_original_policy() -> None:
    original, evidenced, risks = inputs(-0.2)

    target = blend_policy_by_regret(
        original, evidenced, risks, temperature=0.02
    )

    assert dict(target.policy) == dict(original.policy)
    assert target.policy_regret_adjustment.regret == 0.0
    assert target.policy_regret_adjustment.redirect_fraction == 0.0
    assert target.policy_regret_adjustment.redirect_actions == (evidenced.selected_action,)


def test_advantage_continuously_moves_mass_to_safe_policy() -> None:
    original, evidenced, risks = inputs(0.04)

    target = blend_policy_by_regret(
        original, evidenced, risks, temperature=0.02
    )
    adjustment = target.policy_regret_adjustment

    assert adjustment.regret == pytest.approx(0.04)
    assert adjustment.redirect_fraction == pytest.approx(1.0 - math.exp(-2.0))
    assert dict(target.policy)[evidenced.selected_action] > 0.5
    assert dict(target.policy)[1] < 0.5
