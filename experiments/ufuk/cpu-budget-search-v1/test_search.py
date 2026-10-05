"""Rules/history, every-root budget and tactical quiescence correctness."""

import chess
import pytest
from search import BudgetSearch


def material(board):
    values = (0, 1, 3, 3, 5, 9, 0)
    return (
        sum(
            values[p] * (len(board.pieces(p, board.turn)) - len(board.pieces(p, not board.turn)))
            for p in range(1, 7)
        )
        / 40
    )


def test_terminal_mover_and_claim_history():
    mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    result = BudgetSearch(lambda _: pytest.fail("terminal NN query")).search(mate)
    assert result.move is None and result.value == -2 and result.nodes == 1
    board = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        board.push_uci(move)
    result = BudgetSearch(lambda _: pytest.fail("claimed draw NN query")).search(board)
    assert result.move is None and result.value == 0
    # Same FEN without repetition history is not silently treated as this draw.
    result = BudgetSearch(lambda _: 0, nodes=21).search(chess.Board(board.fen()))
    assert result.move is not None


def test_all_root_coverage_and_exact_budget_board_unchanged():
    board = chess.Board()
    before = board.fen(), tuple(board.move_stack)
    seen = []
    result = BudgetSearch(lambda b: seen.append(b.peek()) or 0, nodes=21).search(board)
    assert set(seen) == set(board.legal_moves)
    assert result.nodes == 21 and result.root_actions == 20 and result.completed_depth == 1
    assert (board.fen(), tuple(board.move_stack)) == before
    with pytest.raises(ValueError, match="every legal root"):
        BudgetSearch(lambda _: 0, nodes=20).search(board)


def test_mate_in_one_without_network_knowledge():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    result = BudgetSearch(lambda _: 0, nodes=128).search(board)
    assert board.is_legal(result.move)
    board.push(result.move)
    assert board.is_checkmate() and result.value > 1


def test_quiescence_refutes_poisoned_capture():
    board = chess.Board("rq2k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    assert board.is_valid() and board.is_legal(chess.Move.from_uci("a1a8"))
    shallow = BudgetSearch(material, nodes=board.legal_moves.count() + 1).search(board)
    assert shallow.move.uci() == "a1a8"
    searched = BudgetSearch(material, nodes=4096, quiescence_plies=2, max_depth=1).search(board)
    assert searched.move.uci() != "a1a8" and searched.completed_depth == 1
    assert searched.value >= 0


def test_zero_evaluator_avoids_claiming_strength_and_nonfinite_rejected():
    a = BudgetSearch(lambda _: 0, nodes=512).search(chess.Board())
    b = BudgetSearch(lambda _: 0, nodes=512).search(chess.Board())
    assert a == b and a.nodes <= 512 and a.move is not None
    with pytest.raises(ValueError, match="finite mover"):
        BudgetSearch(lambda _: float("nan")).search(chess.Board())
