import chess

from harbichess.chess.actions import move_to_action
from harbichess.evaluation.teacher_instability import _branching, _segment, _tactical


def test_teacher_instability_segments_branching_and_tacticality() -> None:
    board = chess.Board()
    assert _branching(20) == "low"
    assert _branching(21) == "medium"
    assert _branching(36) == "high"
    assert _tactical(board, move_to_action(board, chess.Move.from_uci("e2e4"))) == "quiet"


def test_teacher_instability_summary_tracks_gate_failures() -> None:
    row = {
        "stable_q_verified_spearman": 0.30,
        "high_q_verified_spearman": 0.32,
        "cross_budget_q_spearman": 0.75,
        "cross_budget_q_drift": 0.02,
        "stable_visit_mass": 0.90,
        "stable_actions": 10,
        "stable_q_spread": 0.20,
        "mean_stable_drift": 0.01,
    }
    summary = _segment((row, {**row, "stable_q_verified_spearman": -0.10}))
    assert summary["positions"] == 2
    assert summary["below_gate_ratio"] == 1.0
    assert summary["negative_rho_ratio"] == 0.5
