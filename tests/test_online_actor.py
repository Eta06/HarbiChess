import copy
import random

import pytest

from harbichess.chess.actions import legal_action_indices, move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.selfplay.online_actor import ActorOpening, OnlineActorConfig, OnlineActors


def actors(*, count=2, cap=3, temperature=1.0, openings=None, rng=None, cursor=None):
    if openings is None:
        rules = PythonChessRules()
        initial = rules.initial_state()
        black = rules.apply(initial, ChessMove("e2e4"))
        openings = (ActorOpening("family-white", initial), ActorOpening("family-black", black))
    return OnlineActors(
        openings,
        config=OnlineActorConfig(count, cap, True, temperature),
        rng=rng or random.Random(719),
        cursor=cursor,
    )


def policies(actor):
    return tuple(
        tuple(
            [1 / len(legal_action_indices(actor.rules.inspect(state)))]
            * len(legal_action_indices(actor.rules.inspect(state)))
        )
        for state in actor.states
    )


def test_complete_game_histories_rng_and_next_rollout_recover_exactly():
    original = actors()
    for _ in range(2):
        original.step(policies(original))
    cursor, rng_state = original.cursor(), original.rng.getstate()
    expected = original.step(policies(original))
    expected_cursor = original.cursor()
    recovered_rng = random.Random(0)
    recovered_rng.setstate(rng_state)
    recovered = actors(rng=recovered_rng, cursor=cursor)
    assert recovered.cursor() == cursor
    assert recovered.rng.getstate() == rng_state  # Restore does not consume RNG.
    assert recovered.step(policies(recovered)) == expected
    assert recovered.cursor() == expected_cursor
    assert recovered.rng.getstate() == original.rng.getstate()
    assert sum(recovered.terminations.values()) == 2
    assert recovered.terminations == {"unknown-rollout-cutoff": 2}
    assert all(row.terminal_result is None and row.rollout_cutoff for row in expected)
    assert all(row.post.moves[:-1] == row.pre.moves for row in expected)


def test_terminal_mate_at_cap_preserves_real_result_and_restarts():
    rules = PythonChessRules()
    state = rules.initial_state("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    actor = actors(count=1, cap=1, openings=(ActorOpening("mate", state),))
    board = actor.rules.inspect(state)
    move = next(move for move in board.legal_moves if move.uci() == "f7g7")
    chosen = move_to_action(board, move)
    policy = tuple(float(index == chosen) for index in legal_action_indices(board))
    (transition,) = actor.step((policy,))
    assert transition.rollout_cutoff is True
    assert transition.terminal_result == "1-0"
    assert transition.terminal_termination == "checkmate"
    assert actor.terminations == {"checkmate": 1}
    assert actor.games[0].game_index == 1 and actor.states == (state,)


def test_explicit_temperature_changes_actual_proposal_probabilities():
    actor = actors(count=1, temperature=0.5)
    count = len(legal_action_indices(actor.rules.inspect(actor.states[0])))
    probabilities = (0.25, 0.75) + (0.0,) * (count - 2)
    (row,) = actor.step((probabilities,))
    assert row.policy_probability in (0.25, 0.75)
    expected = row.policy_probability**2 / (0.25**2 + 0.75**2)
    assert row.behavior_probability == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["nan", "negative", "zero", "unnormalized"])
def test_bad_policy_does_not_change_any_actor_or_rng(kind):
    actor = actors()
    good = policies(actor)
    bad = list(good[1])
    if kind == "nan":
        bad[0] = float("nan")
    elif kind == "negative":
        bad[0] = -0.1
    elif kind == "zero":
        bad = [0.0] * len(bad)
    else:
        bad[0] = 0.5
    before, rng = actor.cursor(), actor.rng.getstate()
    with pytest.raises(ValueError):
        actor.step((good[0], tuple(bad)))
    assert actor.cursor() == before and actor.rng.getstate() == rng


def test_numeric_support_underflow_is_rejected_before_sampling():
    actor = actors(count=1, temperature=5e-324)
    count = len(legal_action_indices(actor.rules.inspect(actor.states[0])))
    policy = (0.25, 0.75) + (0.0,) * (count - 2)
    before, rng = actor.cursor(), actor.rng.getstate()
    with pytest.raises(ValueError, match="underflow"):
        actor.step((policy,))
    assert actor.cursor() == before and actor.rng.getstate() == rng


@pytest.mark.parametrize("what", ["book", "config", "counter", "history", "game-id"])
def test_invalid_restore_is_rejected(what):
    actor = actors()
    cursor = copy.deepcopy(actor.cursor())
    if what == "book":
        cursor["book_sha256"] = "0" * 64
    elif what == "config":
        cursor["config"]["temperature"] = 0.5
    elif what == "counter":
        cursor["terminations"] = {"checkmate": 1}
    elif what == "history":
        cursor["games"][0]["moves"] += ["e2e5"]
    else:
        cursor["games"][0]["game_index"] = cursor["games"][1]["game_index"]
    with pytest.raises(ValueError):
        actors(cursor=cursor)
