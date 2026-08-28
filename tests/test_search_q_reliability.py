from pathlib import Path

import pytest

from harbichess.evaluation.search_q_reliability import (
    SearchQReliabilityConfig,
    _gate,
    _ranks,
    _spearman,
)


def _config() -> SearchQReliabilityConfig:
    return SearchQReliabilityConfig(
        consistency_result=Path("consistency.json"),
        run_result=Path("result.json"),
        train_shard=Path("train.jsonl.gz"),
        validation_shard=Path("validation.jsonl.gz"),
        output_dir=Path("output"),
    )


def test_search_q_rank_correlation_handles_ties_and_common_support() -> None:
    assert _ranks({"a": 1.0, "b": 1.0, "c": 2.0}) == {
        "a": 1.5,
        "b": 1.5,
        "c": 3.0,
    }
    assert _spearman({"a": 1.0, "b": 2.0}, {"a": -2.0, "b": 3.0}) == pytest.approx(1)
    assert _spearman({"a": 1.0, "b": 2.0}, {"a": 3.0, "b": -1.0}) == pytest.approx(-1)
    assert _spearman({"a": 1.0}, {"a": 2.0}) == 0.0


def test_search_q_gate_requires_independent_strength_and_stability() -> None:
    passing = {
        "mean_high_budget_q_verified_spearman": 0.35,
        "top_q_verified_delta_95_interval": (0.001, 0.10),
        "top_q_harmful_ratio": 0.10,
        "mean_top_q_verified_regret": 0.10,
        "top_q_agreement": 0.75,
        "mean_cross_budget_q_spearman": 0.70,
        "mean_top_q_verified_delta_vs_raw": 0.08,
        "mean_top_visit_verified_delta_vs_raw": 0.09,
    }
    assert _gate(passing, _config()) == {"passed": True, "reasons": []}

    failed = {
        **passing,
        "mean_high_budget_q_verified_spearman": 0.2,
        "top_q_harmful_ratio": 0.2,
        "top_q_agreement": 0.5,
    }
    assert _gate(failed, _config())["reasons"] == [
        "800-budget Q/verified correlation is below 0.35",
        "top-Q harmful-action ratio exceeds 10%",
        "512-versus-800 top-Q agreement is below 75%",
    ]
