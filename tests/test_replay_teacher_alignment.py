import math
from pathlib import Path

import pytest

from harbichess.core.state import ChessMove
from harbichess.evaluation.replay_teacher_alignment import (
    ReplayTeacherAlignmentConfig,
    _alignment_gate,
    _argmax,
    _kl,
    _tv,
)


def test_replay_teacher_policy_distances_and_argmax() -> None:
    first = ChessMove("e2e4")
    second = ChessMove("d2d4")
    raw = ((first, 0.75), (second, 0.25))
    teacher = ((first, 0.25), (second, 0.75))

    assert _argmax(raw) == first
    assert _argmax(teacher) == second
    assert _tv(raw, teacher) == pytest.approx(0.5)
    assert _kl(teacher, raw) == pytest.approx(
        0.25 * math.log(0.25 / 0.75) + 0.75 * math.log(0.75 / 0.25)
    )


def test_alignment_gate_applies_frozen_integrity_and_strength_thresholds() -> None:
    config = ReplayTeacherAlignmentConfig(
        run_result=Path("run.json"),
        shard=Path("replay.jsonl.gz"),
        output_dir=Path("alignment"),
    )
    summary = {
        "stored_clean_top_action_agreement": 0.96,
        "mean_stored_clean_tv": 0.04,
        "stored_verified_delta_vs_raw_95_interval": (0.01, 0.10),
        "clean_verified_delta_vs_raw_95_interval": (0.02, 0.11),
    }

    assert _alignment_gate(summary, config)["passed"] is True

    summary["mean_stored_clean_tv"] = 0.06
    gate = _alignment_gate(summary, config)
    assert gate["passed"] is False
    assert gate["reasons"] == ["stored-clean policy TV exceeds the frozen maximum"]


def test_alignment_audit_preserves_failed_upstream_replay_gate() -> None:
    config = ReplayTeacherAlignmentConfig(
        run_result=Path("run.json"),
        shard=Path("replay.jsonl.gz"),
        output_dir=Path("alignment"),
    )
    summary = {
        "stored_clean_top_action_agreement": 1.0,
        "mean_stored_clean_tv": 0.0,
        "stored_verified_delta_vs_raw_95_interval": (0.01, 0.10),
        "clean_verified_delta_vs_raw_95_interval": (0.01, 0.10),
    }

    gate = _alignment_gate(summary, config, replay_qualified=False)

    assert gate["passed"] is False
    assert gate["reasons"] == ["upstream replay coverage gate did not pass"]
