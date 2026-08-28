from pathlib import Path

import pytest

from harbichess.core.state import ChessMove
from harbichess.evaluation.teacher_consistency import (
    TeacherConsistencyConfig,
    _classify_row,
    _gate,
    _policy_comparison,
)


def _config() -> TeacherConsistencyConfig:
    return TeacherConsistencyConfig(
        Path("run.json"),
        Path("train.jsonl.gz"),
        Path("validation.jsonl.gz"),
        Path("output"),
    )


def _budget_rows(*, change_top: bool = False):
    first = ChessMove("e2e4")
    second = ChessMove("d2d4")
    rows = {}
    for budget in (64, 128, 256, 512, 800):
        top = second if change_top and budget == 64 else first
        runner = first if top == second else second
        rows[budget] = {
            "policy": ((top, 0.7), (runner, 0.3)),
            "top_action": top,
            "normalized_visit_margin": 0.4,
        }
    return rows


def test_policy_comparison_reports_consensus_and_distance() -> None:
    first = ((ChessMove("e2e4"), 0.75), (ChessMove("d2d4"), 0.25))
    second = ((ChessMove("e2e4"), 0.60), (ChessMove("d2d4"), 0.40))

    comparison = _policy_comparison(first, second)

    assert comparison["top_action_agreement"]
    assert comparison["tv"] == pytest.approx(0.15)
    assert comparison["forward_kl"] >= 0
    assert comparison["reverse_kl"] >= 0
    assert comparison["jensen_shannon"] >= 0


def test_teacher_consistency_classifies_stable_ambiguous_and_harmful() -> None:
    config = _config()

    stable, stable_reasons = _classify_row(
        _budget_rows(), verified_delta=0.08, config=config
    )
    ambiguous, ambiguous_reasons = _classify_row(
        _budget_rows(change_top=True), verified_delta=0.08, config=config
    )
    harmful, harmful_reasons = _classify_row(
        _budget_rows(), verified_delta=-0.05, config=config
    )

    assert stable == "stable/high-confidence"
    assert not stable_reasons
    assert ambiguous == "budget-sensitive/ambiguous"
    assert "top action changes across budgets" in ambiguous_reasons
    assert harmful == "harmful"
    assert harmful_reasons


def test_consensus_gate_requires_stability_and_verified_intervals() -> None:
    config = _config()
    passing = {
        "stable_ratio": 0.25,
        "harmful_ratio": 0.05,
        "stable_verified_delta_95_interval": (0.01, 0.10),
        "high_budget_top_action_agreement": 0.80,
        "all_verified_delta_95_interval": (0.0, 0.08),
    }

    assert _gate(passing, config)["passed"]
    failed = _gate({**passing, "stable_ratio": 0.10}, config)
    assert not failed["passed"]
    assert "stable target ratio is below 20%" in failed["reasons"]


def test_teacher_consistency_rejects_unsorted_budgets() -> None:
    with pytest.raises(ValueError, match="configuration"):
        TeacherConsistencyConfig(
            Path("run.json"),
            Path("train.jsonl.gz"),
            Path("validation.jsonl.gz"),
            Path("output"),
            budgets=(128, 64, 256),
        )
