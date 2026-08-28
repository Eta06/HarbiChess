from __future__ import annotations

from pathlib import Path

import pytest

from harbichess.chess.actions import move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import Side
from harbichess.evaluation.full_gumbel_targets import (
    FullGumbelTargetConfig,
    _determinism_delta,
    select_stratified_records,
)
from harbichess.replay.schema import ReplayRecord


def _record(game_index: int, outcome: int | None) -> ReplayRecord:
    rules = PythonChessRules()
    state = rules.initial_state()
    board = rules.board(state)
    action = move_to_action(board, board.parse_uci("a2a3"))
    return ReplayRecord(
        game_id=f"game-{game_index}",
        game_index=game_index,
        seed=game_index,
        ply=0,
        root_fen=state.root_fen,
        moves=(),
        side_to_move=Side.WHITE,
        policy=((action, 1.0),),
        selected_action=action,
        root_value=0.0,
        outcome_value=outcome,
        repetition_redirected=False,
    )


def test_stratified_selection_is_deterministic_and_covers_outcomes() -> None:
    rules = PythonChessRules()
    records = tuple(
        _record(index, (-1, 0, 1, None)[index % 4]) for index in range(12)
    )

    first = select_stratified_records(records, count=8, seed=17, rules=rules)
    second = select_stratified_records(records, count=8, seed=17, rules=rules)

    assert first == second
    assert len({record.game_id for record in first}) == 8
    assert {record.outcome_value for record in first} == {-1, 0, 1, None}


def test_determinism_delta_checks_action_visits_value_and_soft_target() -> None:
    row = {
        "identity": "game:0:0",
        "selected_action": "a2a3",
        "root_visits": (("a2a3", 4),),
        "root_value": 0.25,
        "target": (("a2a3", 1.0),),
    }

    assert _determinism_delta(row, dict(row)) == 0.0
    assert _determinism_delta(row, {**row, "root_value": 0.2}) == pytest.approx(0.05)
    assert _determinism_delta(row, {**row, "selected_action": "b2b3"}) == float("inf")


def test_target_config_rejects_an_oversized_audit() -> None:
    with pytest.raises(ValueError, match="audit"):
        FullGumbelTargetConfig(
            output_dir=Path("output"),
            model_path=Path("model"),
            train_shard=Path("train"),
            validation_shard=Path("validation"),
            teacher_qualification_result=Path("teacher"),
            train_positions=2,
            validation_positions=2,
            audit_positions=5,
        )
