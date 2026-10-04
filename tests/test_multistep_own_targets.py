import random
from dataclasses import replace

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.selfplay.online_actor import (
    ActorOpening,
    ActorTransition,
    OnlineActorConfig,
    OnlineActors,
)
from harbichess.training.multistep_own_targets import (
    build_multistep_own_targets,
    required_multistep_bootstrap_indices,
)
from harbichess.training.online_targets import build_one_ply_target

BASELINE = (0.2, 0.3, 0.5)
EMA = (0.7, 0.2, 0.1)


def trajectory(rules, moves, *, initial=None, game=0, cutoff=False):
    state = initial or rules.initial_state()
    rows = []
    for i, move in enumerate(moves):
        post = rules.apply(state, ChessMove(move))
        outcome = rules.outcome(post, claim_draw=False)
        rows.append(ActorTransition(
            0, game, "own-game", state, ChessMove(move), post, 0.2, 0.4,
            cutoff and i == len(moves) - 1,
            outcome.result.value if outcome else None,
            outcome.termination if outcome else None,
        ))
        state = post
    return tuple(rows)


def targets(rules, rows, *, ema=None, **changes):
    horizon = changes.pop("horizon", len(rows))
    if ema is None:
        indices = required_multistep_bootstrap_indices(
            rules, rows, horizon=horizon, claim_draw=False
        )
        ema = dict.fromkeys(indices, EMA)
    return build_multistep_own_targets(
        rules, rows, online_pre_wdl=(BASELINE,) * len(rows),
        ema_leaf_wdl=ema, horizon=horizon, importance_maximum=1.0,
        claim_draw=False, **changes,
    )


@pytest.mark.parametrize("moves,fen,cutoff", [
    (["e2e4"], None, False),
    (["e2e4"], None, True),
    (["f7g7"], "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", True),
    (["f2g2"], "8/8/8/8/8/6k1/5q2/7K b - - 0 1", False),
    (["c1b2"], "7k/8/8/8/8/8/1r6/K1B5 w - - 0 1", False),
])
def test_horizon_one_exact_existing_scalar_equivalence(moves, fen, cutoff):
    rules = PythonChessRules()
    rows = trajectory(rules, moves, initial=rules.initial_state(fen) if fen else None,
                      cutoff=cutoff)
    new = targets(rules, rows)[0]
    row = rows[0]
    old = build_one_ply_target(
        rules, row.pre, row.action, online_pre_wdl=BASELINE,
        ema_post_wdl=None if row.terminal_result else EMA,
        policy_probability=row.policy_probability, behavior_probability=row.behavior_probability,
        importance_maximum=1.0, claim_draw=False, rollout_cutoff=cutoff,
    )
    assert (new.mover_wdl, new.advantage, new.importance, new.target_source,
            new.terminal_result, new.terminal_termination) == (
        old.mover_wdl, old.advantage, old.importance, old.target_source,
        old.terminal_result, old.terminal_termination,
    )
    assert new.credit_plies == 1


@pytest.mark.parametrize("horizon", [8, 16])
def test_fixed_horizon_bootstrap_suffix_parity_and_full_history(horizon):
    rules = PythonChessRules()
    # Pawn moves prevent an accidental repetition/draw in this parity probe.
    moves = ["a2a3", "a7a6", "b2b3", "b7b6", "c2c3", "c7c6", "d2d3", "d7d6",
             "e2e3", "e7e6", "f2f3", "f7f6", "g2g3", "g7g6", "h2h3", "h7h6"]
    rows = trajectory(rules, moves[:horizon])
    result = targets(rules, rows)
    assert [row.credit_plies for row in result] == list(range(horizon, 0, -1))
    assert {row.leaf_index for row in result} == {horizon - 1}
    for row in result:
        assert row.mover_wdl == ((0.1, 0.2, 0.7) if row.credit_plies % 2 else EMA)
        assert row.terminal_result is None
        assert row.transition.post.moves == rows[row.leaf_index].post.moves[
            : row.transition.post.ply
        ]


@pytest.mark.parametrize("moves,expected", [
    (["f2f3", "e7e5", "g2g4", "d8h4"], [(0., 0., 1.), (1., 0., 0.)] * 2),
    (["e2e4", "f7f6", "d2d4", "g7g5", "d1h5"],
     [(1., 0., 0.), (0., 0., 1.), (1., 0., 0.), (0., 0., 1.), (1., 0., 0.)]),
])
def test_observed_black_and_white_mate_credit_all_previous_movers(moves, expected):
    rules = PythonChessRules()
    rows = trajectory(rules, moves, cutoff=True)
    assert required_multistep_bootstrap_indices(
        rules, rows, horizon=len(rows), claim_draw=False
    ) == ()
    result = targets(rules, rows, ema={})
    assert [row.mover_wdl for row in result] == expected
    assert all(row.target_source == "observed-terminal" for row in result)
    assert all(row.leaf_rollout_cutoff for row in result)
    with pytest.raises(ValueError, match="exactly the nonterminal"):
        targets(rules, rows, ema={len(rows) - 1: EMA})


def test_unknown_cap_and_reset_game_never_share_leaf_or_invent_draw():
    rules = PythonChessRules()
    first = trajectory(rules, ["e2e4", "e7e5"], cutoff=True)
    second = trajectory(rules, ["d2d4", "d7d5"], game=1)
    result = targets(rules, first + second, ema={1: EMA, 3: (0.1, 0.1, 0.8)})
    assert [row.leaf_index for row in result] == [1, 1, 3, 3]
    assert [row.credit_plies for row in result] == [2, 1, 2, 1]
    assert [row.mover_wdl for row in result] == [EMA, (0.1, 0.2, 0.7),
                                               (0.1, 0.1, 0.8), (0.8, 0.1, 0.1)]
    assert all(row.terminal_result is None for row in result)
    assert all(row.mover_wdl != (0.0, 1.0, 0.0) for row in result)


def test_terminal_then_reset_inside_block_preserves_observed_terminal_credit():
    rules = PythonChessRules()
    mate = trajectory(rules, ["f2f3", "e7e5", "g2g4", "d8h4"])
    reset = trajectory(rules, ["e2e4", "e7e5"], game=1)
    result = targets(rules, mate + reset, ema={5: EMA})
    assert [row.leaf_index for row in result] == [3, 3, 3, 3, 5, 5]
    assert all(row.terminal_result == "0-1" for row in result[:4])
    assert all(row.terminal_result is None for row in result[4:])
    assert result[0].mover_wdl == (0., 0., 1.)
    assert result[4].mover_wdl == EMA


def test_importance_uses_only_recorded_own_action_and_explicit_cap():
    rules = PythonChessRules()
    rows = trajectory(rules, ["e2e4", "e7e5"], cutoff=True)
    rows = (replace(rows[0], policy_probability=0.9, behavior_probability=0.1), rows[1])
    result = targets(rules, rows)
    assert [row.importance for row in result] == [1.0, 0.5]
    assert all(row.terminal_result is None for row in result)


def test_actual_multi_actor_rng_cursor_are_untouched_by_return_construction():
    rules = PythonChessRules()
    actor = OnlineActors(
        (ActorOpening("root", rules.initial_state()),),
        config=OnlineActorConfig(2, 3, False, 1.0), rng=random.Random(19),
    )
    rows = []
    for _ in range(8):
        policies = tuple(tuple([1 / len(rules.legal_moves(state))] * len(rules.legal_moves(state)))
                         for state in actor.states)
        rows.extend(actor.step(policies))
    cursor, rng = actor.cursor(), actor.rng.getstate()
    result = targets(rules, tuple(rows), horizon=8)
    assert actor.cursor() == cursor and actor.rng.getstate() == rng
    assert len(result) == 16
    assert actor.terminations == {"unknown-rollout-cutoff": 4}
    assert all(row.importance == 1 for row in result)
    assert all(row.credit_plies <= 3 for row in result)
    for row in result:
        leaf = rows[row.leaf_index]
        assert row.transition.slot == leaf.slot
        assert row.transition.game_index == leaf.game_index


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: (rows[0], replace(rows[1], game_index=1)), "reset without closure"),
    (lambda rows: (replace(rows[0], rollout_cutoff=True), rows[1]), "follows closure"),
    (lambda rows: (rows[0], replace(rows[1], source_id="other")), "discontinuous"),
    (lambda rows: (replace(rows[0], post=rows[0].pre), rows[1]), "post-state"),
    (lambda rows: (replace(rows[0], terminal_result="1/2-1/2"), rows[1]), "terminal metadata"),
    (lambda rows: (rows[0], replace(rows[1], slot=1)), "actor-step order"),
])
def test_reject_broken_provenance_instead_of_cross_episode_credit(mutation, match):
    rules = PythonChessRules()
    rows = trajectory(rules, ["e2e4", "e7e5"])
    with pytest.raises(ValueError, match=match):
        targets(rules, mutation(rows))


def test_repetition_claim_uses_full_prefix_and_declared_draw_convention():
    rules = PythonChessRules()
    prefix = trajectory(rules, ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"])
    rows = trajectory(rules, ["f3g1"], initial=prefix[-1].post)
    outcome = rules.outcome(rows[0].post, claim_draw=True)
    claimed = (replace(rows[0], terminal_result=outcome.result.value,
                       terminal_termination=outcome.termination),)
    result = build_multistep_own_targets(
        rules, claimed, online_pre_wdl=(BASELINE,), ema_leaf_wdl={}, horizon=1,
        importance_maximum=1, claim_draw=True,
    )[0]
    assert result.terminal_termination == "threefold_repetition"
    assert result.mover_wdl == (0., 1., 0.)
    assert targets(rules, rows)[0].target_source == "ema-bootstrap"
    assert rules.outcome(rules.initial_state(rules.view(rows[0].post).fen), claim_draw=True) is None


def test_required_leaf_mapping_and_complete_fixed_block_are_explicit():
    rules = PythonChessRules()
    rows = trajectory(rules, ["e2e4", "e7e5"])
    for ema in ({}, {0: EMA, 1: EMA}, {True: EMA}):
        with pytest.raises(ValueError, match="exactly the nonterminal"):
            targets(rules, rows, ema=ema)
    with pytest.raises(ValueError, match="complete actor steps"):
        targets(rules, rows, horizon=3)
    with pytest.raises(ValueError, match="positive integer"):
        targets(rules, rows, horizon=True)
    with pytest.raises(ValueError, match="sum to one"):
        targets(rules, rows, ema={1: (0.1, 0.1, 0.1)})
