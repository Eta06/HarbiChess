import pytest

from harbichess.chess.actions import move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, Side
from harbichess.replay.audit import AuditVerdict, audit_continuation_record
from harbichess.replay.schema import ReplayRecord
from harbichess.search.mcts import MoveStatistics, SearchResult


def _choice_state(rules: PythonChessRules):
    state = rules.initial_state()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        state = rules.apply(state, ChessMove(uci))
    return state


def _record(rules: PythonChessRules) -> ReplayRecord:
    state = _choice_state(rules)
    board = rules.board(state)
    alternative = board.parse_uci("h8g8")
    action = move_to_action(board, alternative)
    return ReplayRecord(
        game_id="continuation-audit",
        game_index=0,
        seed=1,
        ply=state.ply,
        root_fen=state.root_fen,
        moves=tuple(move.uci for move in state.moves),
        side_to_move=Side.BLACK,
        policy=((action, 1.0),),
        selected_action=action,
        root_value=0.0,
        outcome_value=0,
        repetition_redirected=True,
    )


@pytest.mark.parametrize(
    ("alternative_value", "repeat_visits", "target_visits", "expected"),
    [
        (0.20, 4, 12, AuditVerdict.RELIABLE),
        (0.02, 12, 4, AuditVerdict.UNCERTAIN),
        (-0.20, 4, 12, AuditVerdict.HARMFUL),
    ],
)
def test_audit_compares_target_against_repeat_draw_value(
    alternative_value: float,
    repeat_visits: int,
    target_visits: int,
    expected: AuditVerdict,
) -> None:
    rules = PythonChessRules()
    record = _record(rules)
    repeating = ChessMove("f6g8")
    alternative = ChessMove("h8g8")
    search = SearchResult(
        (
            MoveStatistics(repeating, repeat_visits, 0.6, 0.0),
            MoveStatistics(alternative, target_visits, 0.4, alternative_value),
        ),
        0.0,
        16,
    )

    audit = audit_continuation_record(
        record,
        search,
        rules,
        source_run="source",
    )

    assert audit.verdict is expected
    assert audit.repeat_visit_mass == pytest.approx(repeat_visits / 16)
    assert audit.target_visit_mass == pytest.approx(target_visits / 16)
    assert audit.target_mcts_value == pytest.approx(alternative_value)
    assert audit.champion_selected_repeats is (repeat_visits > target_visits)
    assert audit.target_contains_champion_selection is (target_visits > repeat_visits)
