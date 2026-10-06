"""Synthetic tests only; no journal preparation or SF process invocation."""

import chess
import chess.engine
import pytest
from selection import select, teacher_target


def groups():
    return {
        g: [
            dict(
                row_id=f"{g}:{i}",
                trajectory_id=g,
                fen4=f"{g}-position{i}",
                history_sha256=f"{g}-history{i}",
            )
            for i in range(4)
        ]
        for g in ["a", "b"]
    }


def test_selection_balances_and_excludes_actual_state_history_aliases():
    data = groups()
    result = select(
        data, seed=1, count=4, excluded_positions={"a-position0"}, excluded_histories={"b-history1"}
    )
    assert all(r["fen4"] != "a-position0" and r["history_sha256"] != "b-history1" for r in result)
    assert sum(r["trajectory_id"] == "a" for r in result) == 2
    assert result == select(
        data, seed=1, count=4, excluded_positions={"a-position0"}, excluded_histories={"b-history1"}
    )
    with pytest.raises(ValueError):
        select(
            data,
            seed=1,
            count=7,
            excluded_positions={"a-position0"},
            excluded_histories={"b-history1"},
        )


def test_duplicate_fullhistory_aliases_cannot_fill_quota():
    data = groups()
    for rows in data.values():
        for r in rows:
            r["history_sha256"] = "same"
    with pytest.raises(ValueError):
        select(data, seed=1, count=2, excluded_positions=set(), excluded_histories=set())


def test_mover_perspective_cp_proxy_mate_and_ambiguous_zero():
    score = chess.engine.PovScore(chess.engine.Cp(600), chess.WHITE)
    assert teacher_target(score, chess.WHITE)["target"] > 0
    assert teacher_target(score, chess.BLACK)["target"] < 0
    assert (
        teacher_target(chess.engine.PovScore(chess.engine.Mate(-2), chess.WHITE), chess.BLACK)[
            "target"
        ]
        == 1.0
    )
    assert (
        teacher_target(chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE), chess.BLACK)[
            "target"
        ]
        == 0.0
    )
    with pytest.raises(ValueError):
        teacher_target(chess.engine.PovScore(chess.engine.Mate(0), chess.WHITE), chess.WHITE)
