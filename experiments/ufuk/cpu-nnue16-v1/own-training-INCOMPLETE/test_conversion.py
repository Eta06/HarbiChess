"""Synthetic packet/rules tests only; no NN forward, real collection, or optimizer."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
from convert import alias, verify_row


def fixture():
    b = chess.Board()
    row = dict(
        root_fen=chess.STARTING_FEN,
        history_uci=[],
        fen4=" ".join(b.fen().split()[:4]),
        mover="white",
        root_alias=alias(b),
        selected_best_uci="e2e4",
        raw_q_mover=0.2,
        clipped_q_mover=0.2,
        mate_range_score_returned=False,
        behavior_policy_available=False,
        label_source="own-frozen-parent-search",
        root_actions=20,
        nodes=21,
        evaluations=20,
        actual_eval_calls=20,
        completed_depth=0,
    )
    feature = SimpleNamespace(board_indices=lambda _b: [1, 2])
    prior = SimpleNamespace(PRIOR=[1.0], features=lambda _b: [0.1], SCALE=600)
    return row, feature, prior


def test_exact_mover_history_clip_and_prior():
    row, f, p = fixture()
    result = verify_row(row, f, p)
    assert result == dict(indices=[1, 2], prior_logit=0.1 / 600, target=0.2)
    for field, val in [
        ("mover", "black"),
        ("history_uci", ["e2e5"]),
        ("clipped_q_mover", 0.3),
        ("root_actions", 19),
        ("selected_best_uci", "e2e5"),
    ]:
        bad = dict(row)
        bad[field] = val
        with pytest.raises(ValueError):
            verify_row(bad, f, p)


def test_mate_range_is_honest_search_proxy_not_wdl():
    row, f, p = fixture()
    row.update(raw_q_mover=1.2, clipped_q_mover=1.0, mate_range_score_returned=True)
    assert verify_row(row, f, p)["target"] == 1.0
    row["mate_range_score_returned"] = False
    with pytest.raises(ValueError):
        verify_row(row, f, p)


def test_cli_help_no_execution():
    for name in ["convert.py", "prove.py"]:
        r = subprocess.run(
            [sys.executable, str(Path(__file__).with_name(name)), "--help"],
            capture_output=True,
            timeout=10,
        )
        assert r.returncode == 0, r.stderr.decode()
