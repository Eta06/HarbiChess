import chess

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove


def test_cached_claim_flags_and_identical_board_with_different_history():
    rules = PythonChessRules()
    state = rules.initial_state()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
        state = rules.apply(state, ChessMove(uci))
    board = rules.board(state)
    without_history = rules.initial_state(board.fen())
    assert rules.view(state).fen == rules.view(without_history).fen
    for _ in range(3):
        assert rules.outcome(state, claim_draw=False) is None
        assert rules.outcome(state, claim_draw=True).termination == "threefold_repetition"
        assert rules.outcome(without_history, claim_draw=True) is None
        assert rules.board(state).fen() == board.fen()
    assert len(rules.board(state).move_stack) == 7


def test_fact_cache_is_bounded_and_agrees_with_independent_rules():
    rules = PythonChessRules(board_cache_size=2)
    state = rules.initial_state()
    for uci in ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"):
        state = rules.apply(state, ChessMove(uci))
        board = rules.board(state)
        for _ in range(2):
            assert {m.uci for m in rules.legal_moves(state)} == {m.uci() for m in board.legal_moves}
            assert rules.view(state).fen == board.fen()
            for claim in (False, True):
                actual = rules.outcome(state, claim_draw=claim)
                expected = board.outcome(claim_draw=claim)
                assert (actual is None) == (expected is None)
        assert len(rules._thread_local.fact_cache) <= 2
    mate = rules.initial_state("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    assert rules.outcome(mate).termination == "checkmate"
    assert rules.outcome(mate, claim_draw=True).termination == "checkmate"
    assert rules.legal_moves(mate) == ()
    assert chess.Board(rules.view(mate).fen).is_checkmate()
