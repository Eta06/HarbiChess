from pathlib import Path

import pytest

from harbichess.core.state import Side
from harbichess.evaluation.value_signal_audit import (
    ValueSignalAuditConfig,
    _diagnosis,
    _late,
    _position_split,
    _shuffle_game_results,
)
from harbichess.replay.schema import ReplayRecord


def _record(game: str, ply: int, side: Side, outcome: int) -> ReplayRecord:
    return ReplayRecord(
        game_id=game,
        game_index=0,
        seed=1,
        ply=ply,
        root_fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        moves=tuple("a1a2" for _ in range(ply)),
        side_to_move=side,
        policy=((0, 1.0),),
        selected_action=0,
        root_value=0.0,
        outcome_value=outcome,
        repetition_redirected=False,
    )


def _records() -> tuple[ReplayRecord, ...]:
    rows = []
    for game, winner in (("white", 1), ("black", -1), ("draw", 0)):
        for ply in range(8):
            side = Side.WHITE if ply % 2 == 0 else Side.BLACK
            outcome = 0 if winner == 0 else winner * (1 if side is Side.WHITE else -1)
            rows.append(_record(game, ply, side, outcome))
    return tuple(rows)


def test_position_split_is_deterministic_and_stratified() -> None:
    records = _records()

    first = _position_split(records, seed=9)
    second = _position_split(records, seed=9)

    assert first == second
    assert set(record.outcome_value for record in first[0]) == {-1, 0, 1}
    assert set(record.outcome_value for record in first[1]) == {-1, 0, 1}


def test_late_window_keeps_last_rows_per_game() -> None:
    selected = _late(_records(), count=3)

    assert len(selected) == 9
    assert {record.ply for record in selected} == {5, 6, 7}


def test_shuffled_results_remain_internally_perspective_consistent() -> None:
    shuffled = _shuffle_game_results(_records(), seed=4)
    by_game = {}
    for record in shuffled:
        signed = record.outcome_value * (1 if record.side_to_move is Side.WHITE else -1)
        by_game.setdefault(record.game_id, set()).add(signed)

    assert all(len(results) == 1 for results in by_game.values())
    assert sorted(next(iter(results)) for results in by_game.values()) == [-1, 0, 1]


def test_diagnosis_prioritizes_shuffled_leakage_and_position_memorization() -> None:
    failed = {
        "passed": False,
        "selected_step": 0,
        "baseline": {"macro_cross_entropy": 1.1},
        "selected": {
            "validation": {
                "macro_cross_entropy": 1.1,
                "expected_score_pearson": 0.0,
                "loss_draw_margin": 0.0,
                "win_draw_margin": 0.0,
            }
        },
    }
    passed = {**failed, "passed": True}
    memorized = {
        **failed,
        "selected_step": 140,
        "selected": {
            "validation": {
                "macro_cross_entropy": 1.06,
                "expected_score_pearson": 0.21,
                "loss_draw_margin": 0.06,
                "win_draw_margin": 0.04,
            }
        },
    }
    assert (
        _diagnosis(
            {
                "game-disjoint-all": failed,
                "position-split-all": passed,
                "game-disjoint-late32": failed,
                "game-disjoint-shuffled": failed,
            }
        )["verdict"]
        == "insufficient_independent_games"
    )
    assert (
        _diagnosis(
            {
                "game-disjoint-all": failed,
                "position-split-all": failed,
                "game-disjoint-late32": failed,
                "game-disjoint-shuffled": passed,
            }
        )["verdict"]
        == "leakage_or_spurious_game_identity"
    )
    partial = _diagnosis(
        {
            "game-disjoint-all": failed,
            "position-split-all": memorized,
            "game-disjoint-late32": failed,
            "game-disjoint-shuffled": failed,
        }
    )
    assert partial["verdict"] == "position_memorization_without_game_generalization"
    assert partial["position_split_partial_memorization"] is True


def test_value_signal_config_requires_aligned_steps() -> None:
    with pytest.raises(ValueError, match="schedule"):
        ValueSignalAuditConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            train_shard=Path("train"),
            validation_shard=Path("validation"),
            steps=21,
            validation_interval=20,
        )
