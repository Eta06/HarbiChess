import chess
import pytest
from value import PRIOR, ClassicalValue, features


def test_initial_balanced_color_reflection_and_board_unchanged():
    b = chess.Board()
    old = b.fen()
    assert ClassicalValue()(b) == 0.0
    b.push_uci("e2e4")
    b.push_uci("e7e5")
    b.push_uci("g1f3")
    snapshot = (b.fen(), tuple(b.move_stack))
    assert features(b) == pytest.approx(features(b.mirror()), abs=1e-12)
    assert ClassicalValue()(b) == pytest.approx(ClassicalValue()(b.mirror()), abs=1e-12)
    assert (b.fen(), tuple(b.move_stack)) == snapshot and old == chess.STARTING_FEN


def test_material_mover_sign_and_weights_not_search_only():
    b = chess.Board("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1")
    a = ClassicalValue()(b)
    b.turn = chess.BLACK
    assert a > 0 and ClassicalValue()(b) == pytest.approx(-a)
    w = list(PRIOR)
    w[4] = 0.0
    assert ClassicalValue(w)(b) != ClassicalValue()(b)


def test_exact_mate_stalemate_insufficient_and_fullhistory_claim():
    assert ClassicalValue()(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")) == -1.0
    assert ClassicalValue()(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == 0.0
    assert ClassicalValue()(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")) == 0.0
    b = chess.Board()
    for u in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        b.push_uci(u)
    assert b.outcome(claim_draw=True) is not None and ClassicalValue()(b) == 0.0
    assert len(b.move_stack) == 8


def test_bad_weight_rejected():
    with pytest.raises(ValueError):
        ClassicalValue((0.0,))
    with pytest.raises(ValueError):
        ClassicalValue((float("nan"),) * len(PRIOR))


def test_zero_residual_exact_prior_and_nonzero_changes_mover_scalar():
    b = chess.Board("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1")
    assert ClassicalValue(theta=[0.0] * 18)(b) == ClassicalValue()(b)
    theta = [0.0] * 18
    theta[4] = -0.4
    assert ClassicalValue(theta=theta)(b) < ClassicalValue()(b)
    a = ClassicalValue(theta=theta)(b)
    b.turn = chess.BLACK
    assert ClassicalValue(theta=theta)(b) == pytest.approx(-a)
