import math

import pytest

from harbichess.chess.rules import IllegalMoveError, PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.training.online_targets import (
    ONLINE_TARGET_SCHEMA,
    build_one_ply_target,
    clipped_importance,
    expected_score,
    flip_wdl,
    validate_wdl,
)


def target(rules, pre, uci, **changes):
    arguments = dict(
        online_pre_wdl=(0.2, 0.3, 0.5),
        ema_post_wdl=(0.7, 0.2, 0.1),
        policy_probability=0.2,
        behavior_probability=0.4,
        importance_maximum=1.0,
        claim_draw=True,
        rollout_cutoff=False,
    )
    return build_one_ply_target(rules, pre, ChessMove(uci), **(arguments | changes))


@pytest.mark.parametrize("cutoff", [False, True])
def test_nonterminal_and_cutoff_flip_next_player_wdl_without_false_draw(cutoff):
    rules = PythonChessRules()
    pre = rules.initial_state()
    row = target(rules, pre, "e2e4", rollout_cutoff=cutoff)
    assert row.schema == ONLINE_TARGET_SCHEMA
    assert row.post.moves == (ChessMove("e2e4"),)
    assert row.pre == pre and row.claim_draw is True
    assert row.rollout_cutoff is cutoff
    assert row.target_source == "ema-bootstrap"
    assert row.terminal_result is row.terminal_termination is None
    assert row.mover_wdl == (0.1, 0.2, 0.7)
    assert row.advantage == pytest.approx(-0.15)
    assert row.importance == 0.5
    # E=W+D/2=(Q+1)/2. The policy-gradient scale is explicit.
    pre_q = 0.2 - 0.5
    child_q = 0.7 - 0.1
    assert row.advantage == pytest.approx((-child_q - pre_q) / 2)


@pytest.mark.parametrize(
    ("fen", "move", "result"),
    [
        ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "f7g7", "1-0"),
        ("8/8/8/8/8/6k1/5q2/7K b - - 0 1", "f2g2", "0-1"),
    ],
)
def test_real_mate_uses_movers_win_even_when_cutoff_coincides(fen, move, result):
    rules = PythonChessRules()
    row = target(rules, rules.initial_state(fen), move, ema_post_wdl=None, rollout_cutoff=True)
    assert row.target_source == "observed-terminal"
    assert row.terminal_result == result
    assert row.terminal_termination == "checkmate"
    assert row.mover_wdl == (1.0, 0.0, 0.0)
    assert row.advantage == pytest.approx(0.65)


def test_real_insufficient_material_draw_requires_a_terminal_child():
    rules = PythonChessRules()
    pre = rules.initial_state("7k/8/8/8/8/8/1r6/K1B5 w - - 0 1")
    row = target(rules, pre, "c1b2", ema_post_wdl=None)
    assert row.terminal_result == "1/2-1/2"
    assert row.terminal_termination == "insufficient_material"
    assert row.mover_wdl == (0.0, 1.0, 0.0)
    assert row.advantage == pytest.approx(0.15)


def test_announced_repetition_claim_preserves_history_and_declared_convention():
    rules = PythonChessRules()
    pre = rules.initial_state()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]:
        pre = rules.apply(pre, ChessMove(move))
    claimed = target(rules, pre, "f3g1", ema_post_wdl=None)
    unclaimed = target(rules, pre, "f3g1", claim_draw=False)
    assert claimed.terminal_termination == "threefold_repetition"
    assert claimed.mover_wdl == (0.0, 1.0, 0.0)
    assert unclaimed.target_source == "ema-bootstrap"
    assert len(claimed.post.moves) == 7
    assert rules.outcome(rules.initial_state(rules.view(claimed.post).fen), claim_draw=True) is None
    with pytest.raises(ValueError, match="after a terminal"):
        target(rules, claimed.post, "f6g8")


def test_announced_fifty_move_claim_is_distinct_from_automatic_terminal():
    rules = PythonChessRules()
    pre = rules.initial_state("7k/8/8/8/8/8/8/KR6 w - - 98 1")
    claimed = target(rules, pre, "b1b2", ema_post_wdl=None)
    unclaimed = target(rules, pre, "b1b2", claim_draw=False)
    assert claimed.terminal_termination == "fifty_moves"
    assert unclaimed.target_source == "ema-bootstrap"


def test_reject_illegal_action_missing_bootstrap_and_ambiguous_terminal_label():
    rules = PythonChessRules()
    with pytest.raises(IllegalMoveError):
        target(rules, rules.initial_state(), "e2e5")
    with pytest.raises(ValueError, match="requires explicit EMA"):
        target(rules, rules.initial_state(), "e2e4", ema_post_wdl=None)
    with pytest.raises(ValueError, match="not an EMA"):
        target(rules, rules.initial_state("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"), "f7g7")
    with pytest.raises(ValueError, match="explicit booleans"):
        target(rules, rules.initial_state(), "e2e4", rollout_cutoff=1)


@pytest.mark.parametrize(
    "values",
    [(0.2, 0.3), (0.2, 0.2, 0.2), (1.1, 0.0, -0.1), (math.nan, 0.5, 0.5)],
)
def test_reject_invalid_distributions_without_renormalization(values):
    with pytest.raises(ValueError):
        validate_wdl(values)


def test_wdl_flip_is_involution_and_draws_preserve_score():
    probabilities = (0.17, 0.26, 0.57)
    assert flip_wdl(flip_wdl(probabilities)) == probabilities
    assert expected_score(probabilities) + expected_score(flip_wdl(probabilities)) == 1.0
    assert flip_wdl((0.0, 1.0, 0.0)) == (0.0, 1.0, 0.0)


@pytest.mark.parametrize(
    ("policy", "behavior", "maximum"),
    [(0.3, 0.0, 1.0), (-0.1, 0.2, 1.0), (0.3, 1.1, 1.0), (0.3, 0.2, math.inf)],
)
def test_importance_rejects_invalid_recorded_probabilities(policy, behavior, maximum):
    with pytest.raises(ValueError):
        clipped_importance(policy, behavior, maximum=maximum)


def test_importance_is_explicit_clipped_and_stable_for_subnormal_support():
    assert clipped_importance(0.2, 0.4, maximum=1.0) == 0.5
    assert clipped_importance(0.4, 0.2, maximum=1.0) == 1.0
    assert clipped_importance(0.4, 0.2, maximum=3.0) == 2.0
    assert clipped_importance(1.0, 5e-324, maximum=1.0) == 1.0
    assert clipped_importance(0.0, 5e-324, maximum=1e-100) == 0.0
