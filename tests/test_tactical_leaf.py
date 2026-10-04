import math

import chess
import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.tactical_leaf import (
    MaterialQuiescence,
    TacticalLeafEvaluator,
    material_value,
)


def test_quiescence_checks_terminal_stm_signs_draw_history_and_rule50():
    search = MaterialQuiescence()
    mate = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        mate.push_uci(move)
    assert search.evaluate(mate) == -1
    stalemate = chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    assert search.evaluate(stalemate) == 0
    repeated = chess.Board()
    for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        repeated.push_uci(move)
    assert repeated.can_claim_threefold_repetition() and search.evaluate(repeated) == 0
    fifty = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 100 70")
    assert fifty.can_claim_fifty_moves() and search.evaluate(fifty) == 0
    material = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    before = material_value(material)
    material.turn = chess.BLACK
    assert material_value(material) == -before and 0 < before < 1


def test_quiescence_searches_legal_check_evasions_capture_recapture_ep_and_promotion():
    # In check, White can capture the undefended checking queen; no stand pat.
    board = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
    assert board.is_check() and material_value(board) < 0
    search = MaterialQuiescence(depth=4, node_limit=64)
    before = board.fen(), tuple(board.move_stack)
    assert search.evaluate(board) == 0
    assert search.last_nodes > 1 and (board.fen(), tuple(board.move_stack)) == before
    # Queen can win a pawn, but the rook recaptures: do not value the first capture alone.
    board = chess.Board("4k3/8/8/8/8/3r4/3p4/3QK3 w - - 0 1")
    score = search.evaluate(board)
    assert score == pytest.approx(material_value(board))
    assert search.last_nodes > 1
    ep = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert search.evaluate(ep) > material_value(ep)
    promotion = chess.Board("8/P3k3/8/8/8/8/8/4K3 w - - 0 1")
    assert search.evaluate(promotion) > material_value(promotion)
    assert promotion.piece_at(chess.A7).piece_type == chess.PAWN


def test_quiescence_hard_budget_checked_fallback_and_state_restoration():
    board = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
    state = board.fen(), tuple(board.move_stack)
    truncated = MaterialQuiescence(depth=4, node_limit=1)
    assert truncated.evaluate(board) == material_value(board)
    assert truncated.last_nodes == 1 and truncated.counters.checked_fallback_calls == 1
    assert truncated.counters.node_truncations == 1
    assert (board.fen(), tuple(board.move_stack)) == state
    checked_child = chess.Board("8/P3k3/8/8/8/8/8/4K3 w - - 0 1")
    limited = MaterialQuiescence(depth=1, node_limit=64)
    assert math.isfinite(limited.evaluate(checked_child)) and limited.last_nodes <= 64
    assert limited.counters.depth_truncations > 0


def test_tactical_evaluator_preserves_framework_independent_priors_and_history():
    rules = PythonChessRules()
    state = ChessState(chess.STARTING_FEN, (ChessMove("e2e4"), ChessMove("d7d5")))
    priors = tuple((move, 1 / len(rules.legal_moves(state))) for move in rules.legal_moves(state))
    before = rules.board(state).fen(), tuple(rules.board(state).move_stack)

    class Fixed:
        def evaluate(self, observed):
            assert observed == state
            return PositionEvaluation(priors, -0.5)

    for mode in ("static", "quiescent"):
        evaluator = TacticalLeafEvaluator(Fixed(), rules=rules, mode=mode)
        result = evaluator.evaluate(state)
        assert result.priors is priors and -1 <= result.value <= 1
        assert (rules.board(state).fen(), tuple(rules.board(state).move_stack)) == before
    with pytest.raises(ValueError):
        TacticalLeafEvaluator(Fixed(), mode="unknown")
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            MaterialQuiescence(node_limit=invalid)
        with pytest.raises(ValueError):
            MaterialQuiescence(depth=invalid)
