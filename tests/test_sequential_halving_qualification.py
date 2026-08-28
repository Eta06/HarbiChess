from pathlib import Path

from harbichess.evaluation.sequential_halving_qualification import (
    SequentialHalvingQualificationConfig,
    _gate,
)


def _config() -> SequentialHalvingQualificationConfig:
    return SequentialHalvingQualificationConfig(
        consistency_result=Path("consistency.json"),
        q_reliability_result=Path("q.json"),
        run_result=Path("result.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_sequential_halving_gate_requires_strength_stability_and_exact_budget() -> None:
    passing = {
        "selected_action_agreement": 0.75,
        "verified_delta_95_interval": (0.001, 0.1),
        "harmful_ratio": 0.10,
        "mean_verified_regret": 0.10,
        "mean_verified_delta_vs_raw": 0.08,
        "best_action_coverage": 0.80,
        "all_budgets_exact": True,
    }
    assert _gate(passing, standard_q_delta=0.09, config=_config()) == {
        "passed": True,
        "reasons": [],
    }

    failed = {
        **passing,
        "selected_action_agreement": 0.70,
        "mean_verified_delta_vs_raw": 0.07,
        "all_budgets_exact": False,
    }
    assert _gate(failed, standard_q_delta=0.09, config=_config())["reasons"] == [
        "512-versus-800 selected-action agreement is below 75%",
        "selected-action delta regresses standard top-Q by more than 0.01",
        "sequential-halving evaluation-slot budget was not exact",
    ]
