import math

import chess
import pytest

from harbichess.search.all_legal import AllLegalSearch
from harbichess.search.tactical_leaf import material_value


def test_all_legal_complete_root_finds_mate_and_terminal_history_is_exact():
    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1")
    before = board.fen(), tuple(board.move_stack)
    result = AllLegalSearch(material_value, max_depth=1).search(board, wall_seconds=5)
    assert result.completed_depth == 1 and result.value == 1
    assert result.root_moves_at_completed_depth == len(list(board.legal_moves))
    assert (board.fen(), tuple(board.move_stack)) == before
    board.push(result.move)
    assert board.is_checkmate()
    terminal = AllLegalSearch(material_value).search(board, wall_seconds=1)
    assert terminal.move is None and terminal.value == -1 and terminal.evaluations == 0
    repeated = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        repeated.push_uci(move)
    fifty = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 100 70")
    stalemate = chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    for drawn in (repeated, fifty, stalemate):
        result = AllLegalSearch(material_value).search(drawn, wall_seconds=1)
        assert result.move is None and result.value == 0 and result.evaluations == 0


def test_node_abort_retains_only_completed_iteration_and_legal_fallback():
    board = chess.Board()
    completed = AllLegalSearch(material_value, max_depth=1).search(board, wall_seconds=5)
    assert completed.completed_depth == 1
    partial = AllLegalSearch(material_value, node_limit=completed.nodes + completed.qnodes + 1)
    result = partial.search(board, wall_seconds=5)
    assert result.abort_reason == "nodes" and result.completed_depth == 1
    assert result.move == completed.move and result.value == completed.value
    assert result.nodes + result.qnodes <= partial.node_limit
    fallback = AllLegalSearch(material_value, node_limit=1).search(board, wall_seconds=5)
    assert fallback.abort_reason == "nodes" and fallback.completed_depth == 0
    assert fallback.move in board.legal_moves and fallback.value is None


def test_check_evasions_special_moves_no_checked_stand_pat_and_source_restoration():
    def value(board):
        assert not board.is_check()
        return material_value(board)

    for board in (
        chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1"),
        chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
        chess.Board("8/P3k3/8/8/8/8/8/4K3 w - - 0 1"),
    ):
        before = board.fen(), tuple(board.move_stack)
        result = AllLegalSearch(value, max_depth=1).search(board, wall_seconds=5)
        assert result.move in board.legal_moves and result.completed_depth == 1
        assert (board.fen(), tuple(board.move_stack)) == before
    # Qxc3 removes the checking queen; ...Rxc3 checks again beyond qdepth1.
    checked_chain = chess.Board("2k5/2r5/8/8/8/2q1K3/2Q5/8 w - - 0 1")
    truncated = AllLegalSearch(value, max_depth=1, qdepth=1).search(checked_chain, wall_seconds=5)
    assert truncated.abort_reason == "checked-qdepth"
    assert truncated.completed_depth == 0 and truncated.value is None


def test_wall_abort_and_invalid_evaluator_restore_history(monkeypatch):
    board = chess.Board()
    board.push_uci("e2e4")
    before = board.fen(), tuple(board.move_stack)
    ticks = iter((0.0, 2.0, 3.0))
    with monkeypatch.context() as patch:
        patch.setattr("harbichess.search.all_legal.time.perf_counter", lambda: next(ticks))
        result = AllLegalSearch(material_value).search(board, wall_seconds=1)
    assert result.abort_reason == "wall" and result.completed_depth == 0
    assert result.move in board.legal_moves and result.value is None
    with pytest.raises(ValueError, match="finite STM"):
        AllLegalSearch(lambda _: math.nan).search(board, wall_seconds=1)
    assert (board.fen(), tuple(board.move_stack)) == before
    for budget in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError):
            AllLegalSearch(material_value).search(board, wall_seconds=budget)
