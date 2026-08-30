import pytest

from harbichess.evaluation.cumulative_value_gate import (
    CumulativeGateConfig,
    PredictionGame,
    evaluate_cumulative_gate,
    paired_bootstrap,
    paired_power_plan,
)


def _game(game_id: str, baseline, candidate) -> PredictionGame:
    return PredictionGame(
        game_id=game_id,
        outcomes=(1, 0, -1),
        baseline_probabilities=baseline,
        candidate_probabilities=candidate,
    )


def test_prediction_game_rejects_unpaired_probabilities() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        PredictionGame("bad", (1,), ((1.0, 0.0, 0.0),), ())


def test_paired_bootstrap_preserves_zero_difference() -> None:
    probabilities = ((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8))
    games = tuple(_game(str(index), probabilities, probabilities) for index in range(4))
    result = paired_bootstrap(
        games,
        improvement=False,
        config=CumulativeGateConfig(bootstrap_samples=100, seed=7),
    )

    assert all(
        interval == {"estimate": 0.0, "low": 0.0, "high": 0.0}
        for interval in result["intervals"].values()
    )


def test_cumulative_gate_requires_old_noninferiority_and_fresh_superiority() -> None:
    baseline = ((0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9))
    better = ((0.95, 0.025, 0.025), (0.025, 0.95, 0.025), (0.025, 0.025, 0.95))
    old = tuple(_game(f"old-{index}", baseline, baseline) for index in range(6))
    fresh = tuple(_game(f"fresh-{index}", baseline, better) for index in range(6))
    result = evaluate_cumulative_gate(
        old,
        fresh,
        config=CumulativeGateConfig(bootstrap_samples=100, seed=11),
    )

    assert result["passed"]
    assert all(result["checks"].values())


def test_power_plan_is_deterministic_inflated_and_rounded() -> None:
    plan = paired_power_plan(
        standard_deviation=0.02,
        null_boundary=0.003,
        assumed_effect=0.0,
        inflation=0.15,
        round_to=24,
    )

    assert plan.raw_games > 0
    assert plan.inflated_games >= plan.raw_games
    assert plan.rounded_games >= plan.inflated_games
    assert plan.rounded_games % 24 == 0


def test_power_plan_rejects_zero_effect_gap() -> None:
    with pytest.raises(ValueError, match="differ"):
        paired_power_plan(
            standard_deviation=0.02,
            null_boundary=0.003,
            assumed_effect=0.003,
        )
