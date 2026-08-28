from pathlib import Path

from harbichess.evaluation.policy_candidate_validation import (
    PolicyCandidateValidationConfig,
)


def test_candidate_validation_freezes_fresh_gate_and_tactical_budgets() -> None:
    config = PolicyCandidateValidationConfig(
        convergence_result=Path("convergence.json"),
        policy_target_result=Path("targets.json"),
        dataset_result=Path("dataset.json"),
        run_result=Path("run.json"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
    )
    assert config.minimum_gap_fraction == 0.20
    assert config.maximum_harmful_ratio == 0.10
    assert config.tactical_budgets == (64, 512)
    assert config.seed == 2026082851
