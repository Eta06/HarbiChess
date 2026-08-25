from harbichess.evaluation.paired_gate import (
    PairedGateConfig,
    PairedObservation,
    evaluate_paired_gate,
)


def test_paired_gate_requires_strength_and_behavior_guardrails() -> None:
    settings = PairedGateConfig(bootstrap_samples=1_000, seed=7)
    improving = tuple(
        PairedObservation(
            control_score=0.0,
            candidate_score=1.0,
            control_avoidable=False,
            candidate_avoidable=False,
        )
        for _ in range(20)
    )

    result = evaluate_paired_gate(improving, config=settings)

    assert result.strength_passed
    assert result.avoidable_passed
    assert result.win_rate_passed
    assert result.decisive_passed
    assert result.passed

    repetition_regression = tuple(
        PairedObservation(
            control_score=item.control_score,
            candidate_score=item.candidate_score,
            control_avoidable=False,
            candidate_avoidable=True,
        )
        for item in improving
    )
    regressed = evaluate_paired_gate(repetition_regression, config=settings)

    assert regressed.strength_passed
    assert not regressed.avoidable_passed
    assert not regressed.passed


def test_paired_gate_rejects_draw_only_strength_without_more_wins() -> None:
    settings = PairedGateConfig(bootstrap_samples=1_000, seed=11)
    observations = tuple(
        PairedObservation(
            control_score=0.0,
            candidate_score=0.5,
            control_avoidable=False,
            candidate_avoidable=False,
        )
        for _ in range(20)
    )

    result = evaluate_paired_gate(observations, config=settings)

    assert result.strength_passed
    assert not result.win_rate_passed
    assert not result.passed
