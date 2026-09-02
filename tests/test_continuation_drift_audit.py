import math

from harbichess.evaluation.continuation_drift_audit import _expected_scores, _summarize


def test_expected_scores_use_side_to_move_child_perspective() -> None:
    scores = _expected_scores(
        [
            [math.log(0.7), math.log(0.2), math.log(0.1)],
            [math.log(0.1), math.log(0.2), math.log(0.7)],
        ]
    )

    assert math.isclose(scores[0], -0.6, abs_tol=1e-7)
    assert math.isclose(scores[1], 0.6, abs_tol=1e-7)


def test_summary_preserves_rows_for_paired_comparison() -> None:
    rows = (
        {
            "candidate_spearman": 0.2,
            "candidate_verified_top": True,
        },
        {
            "candidate_spearman": 0.4,
            "candidate_verified_top": False,
        },
    )

    summary = _summarize(rows)

    assert summary["positions"] == 2
    assert math.isclose(summary["candidate_mean_spearman"], 0.3)
    assert summary["candidate_verified_top_agreement"] == 0.5
    assert summary["rows"] == rows
