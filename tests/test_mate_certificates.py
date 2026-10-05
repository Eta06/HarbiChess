import math
import random
from types import SimpleNamespace

import chess
import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.full_gumbel import FullGumbelConfig
from harbichess.search.mate_certificates import (
    certified_policy,
    immediate_mating_moves,
    shield_visited_losses,
    visited_mate_in_one_losses,
)
from harbichess.search.ownsearch_wavefront import WavefrontGumbel


class FixedEvaluator:
    def __init__(self, rules, priors):
        self.rules = rules
        self.priors = priors
        self.calls = 0

    def evaluate_many(self, states):
        self.calls += len(states)
        out = []
        for state in states:
            moves = self.rules.legal_moves(state)
            weights = [self.priors.get(move.uci, 1.0) for move in moves]
            total = sum(weights)
            out.append(
                PositionEvaluation(
                    tuple(
                        (move, weight / total)
                        for move, weight in zip(moves, weights, strict=True)
                    ),
                    -0.37,
                )
            )
        return tuple(out)


@pytest.mark.parametrize(
    ("fen", "mover_is_white"),
    [
        ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", True),
        ("8/8/8/8/8/6k1/5q2/7K b - - 0 1", False),
    ],
)
def test_exact_mate_solver_correct_mover_value_full_support_and_fixed_budget(fen, mover_is_white):
    rules = PythonChessRules()
    state = ChessState(fen)
    mates = immediate_mating_moves(rules, state)
    assert mates
    board = rules.inspect(state)
    legal = tuple(
        ChessMove(move.uci())
        for move in sorted(board.legal_moves, key=lambda m: m.uci())
    )
    # Force exact mates to have the lowest network priors.
    priors = {move.uci: (1e-9 if move in mates else 1.0) for move in legal}
    evaluator = FixedEvaluator(rules, priors)
    config = FullGumbelConfig(16, 4, 0.0, 0.1, 50.0, True)
    result = WavefrontGumbel(evaluator, rules, config).search_many(
        [state], [random.Random(11)]
    )[0]

    assert tuple(m.uci for m in result.certified_mates) == tuple(m.uci for m in mates)
    assert result.selected_action in mates
    assert result.root_value == pytest.approx(1.0)
    assert result.simulations == config.simulations
    by_move = {item.move: item for item in result.moves}
    assert sum(item.visits for item in result.moves) == 16
    assert by_move[result.selected_action].visits == 16
    assert by_move[result.selected_action].mean_value == pytest.approx(1.0)
    weights = dict(result.action_weights)
    assert set(weights) == set(legal)
    assert all(weight > 0 for weight in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert sum(weights[m] for m in mates) > 1 - 1e-9
    assert sum(weights[m] for m in legal if m not in mates) <= 1e-9
    assert evaluator.calls == 1
    assert rules.view(state).side_to_move.value == ("white" if mover_is_white else "black")


def test_all_exact_mates_are_reported_and_multi_mate_policy_never_leaves_wins():
    rules = PythonChessRules()
    state = ChessState("7K/8/7k/1q6/8/8/8/8 b - - 0 1")
    mates = immediate_mating_moves(rules, state)
    assert {m.uci for m in mates} == {"b5e8", "b5b8"}
    legal = tuple(m for m in rules.legal_moves(state))
    probs, selected = certified_policy(
        legal,
        mates,
        {m: (2.0 if m.uci == "b5e8" else 1.0) for m in mates},
    )
    assert selected.uci == "b5e8"
    assert len(probs) == len(legal)
    assert all(p > 0 for p in probs)
    assert sum(p for move, p in zip(legal, probs, strict=True) if move in mates) > 1 - 1e-9
    evaluator = FixedEvaluator(rules, {"b5e8": 2.0, "b5b8": 1.0})
    result = WavefrontGumbel(
        evaluator, rules, FullGumbelConfig(16, 4, 0.0, 0.1, 50.0, True)
    ).search_many([state], [random.Random(19)])[0]
    assert result.certified_mates == mates
    assert result.selected_action in mates
    assert sum(dict(result.action_weights)[m] for m in mates) > 1 - 1e-9
    assert sum(x.visits for x in result.moves) == 16


def test_stalemate_draw_and_claimable_root_do_not_create_mate_certificates():
    rules = PythonChessRules()
    stalemate_move_root = ChessState("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    mates = immediate_mating_moves(rules, stalemate_move_root)
    board = rules.inspect(stalemate_move_root)
    for move in mates:
        post = rules.apply(stalemate_move_root, move)
        assert rules.outcome(post, claim_draw=True).termination == "checkmate"
    for move in board.legal_moves:
        post = rules.apply(stalemate_move_root, ChessMove(move.uci()))
        outcome = rules.outcome(post, claim_draw=True)
        if outcome is not None and outcome.termination == "stalemate":
            assert ChessMove(move.uci()) not in mates

    repeated = ChessState(
        chess.STARTING_FEN,
        tuple(ChessMove(m) for m in ("g1f3", "g8f6", "f3g1", "f6g8") * 2),
    )
    assert rules.outcome(repeated, claim_draw=True) is not None
    assert immediate_mating_moves(rules, repeated, claim_draw=True) == ()


def test_certificate_scan_preserves_cached_board_and_full_history():
    rules = PythonChessRules()
    state = ChessState("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    before = rules.inspect(state).fen(), tuple(m.uci() for m in rules.inspect(state).move_stack)
    first = immediate_mating_moves(rules, state)
    after = rules.inspect(state).fen(), tuple(m.uci() for m in rules.inspect(state).move_stack)
    assert first
    assert immediate_mating_moves(rules, state) == first
    assert before == after


def test_nonterminal_nonmate_root_keeps_existing_search_schedule():
    rules = PythonChessRules()
    state = ChessState(chess.STARTING_FEN)
    assert immediate_mating_moves(rules, state) == ()
    evaluator = FixedEvaluator(rules, {})
    result = WavefrontGumbel(
        evaluator, rules, FullGumbelConfig(16, 4, 0.0, 0.1, 50.0, True)
    ).search_many([state], [random.Random(17)])[0]
    assert result.certified_mates == ()
    assert result.simulations == 16
    assert sum(x.visits for x in result.moves) == 16
    assert len(result.action_weights) == len(rules.legal_moves(state))


def test_visited_loss_certificate_checks_only_visited_nonterminal_children_and_shields_support():
    rules = PythonChessRules()
    state = ChessState("6rk/8/8/8/8/6q1/6B1/6K1 w - - 0 1")
    legal = tuple(rules.legal_moves(state))
    safe, losing = ChessMove("g1f1"), ChessMove("g1h1")
    assert set(legal) == {safe, losing}
    loss_proofs = visited_mate_in_one_losses(rules, state, (losing,))
    assert loss_proofs == ((losing, (ChessMove("g3g2"),)),)
    assert visited_mate_in_one_losses(rules, state, (safe,)) == ()

    stats = (
        SimpleNamespace(move=safe, visits=4, mean_value=-0.05, prior=0.1),
        SimpleNamespace(move=losing, visits=12, mean_value=-0.99, prior=0.9),
    )
    original, selected, status = shield_visited_losses(
        legal,
        (0.25, 0.75),
        stats,
        (losing,),
        selected_action=losing,
    )
    by_move = dict(zip(legal, original, strict=True))
    assert by_move[losing] == 1e-12
    assert by_move[safe] == pytest.approx(1 - 1e-12)
    assert selected == safe
    assert status == "visited-losses-epsilon-shielded"
    assert math.fsum(original) == pytest.approx(1.0, abs=1e-15)
    assert all(p > 0 for p in original)


def test_visited_loss_certificates_skip_terminal_children_and_claimable_roots():
    rules = PythonChessRules()
    # Capturing the last rook leaves K+B versus K: an exact child draw, not a loss.
    state = ChessState("1r5k/2B5/8/8/8/8/8/K7 w - - 0 1")
    capture = ChessMove("c7b8")
    child = rules.apply(state, capture)
    assert rules.outcome(state, claim_draw=True) is None
    assert rules.outcome(child, claim_draw=True).result.value == "1/2-1/2"
    assert visited_mate_in_one_losses(rules, state, (capture,)) == ()

    repeated = ChessState(
        chess.STARTING_FEN,
        tuple(ChessMove(m) for m in ("g1f3", "g8f6", "f3g1", "f6g8") * 2),
    )
    assert rules.outcome(repeated, claim_draw=True) is not None
    assert visited_mate_in_one_losses(
        rules, repeated, tuple(rules.legal_moves(repeated)), claim_draw=True
    ) == ()


def test_visited_loss_shield_win_precedence_all_loss_and_unknown_fallback_ties():
    rules = PythonChessRules()
    legal = tuple(rules.legal_moves(ChessState(chess.STARTING_FEN)))
    weights = tuple(1 / len(legal) for _ in legal)
    first, second = legal[:2]
    stats = tuple(
        SimpleNamespace(
            move=move,
            visits=8 if move in (first, second) else 0,
            mean_value=0.2,
            prior=0.3 if move == first else (0.2 if move == second else 0.0),
        )
        for move in legal
    )
    unchanged, selected, status = shield_visited_losses(
        legal,
        weights,
        stats,
        (first,),
        selected_action=second,
        certified_wins=(second,),
    )
    assert unchanged == weights
    assert selected == second
    assert status == "winning-certificate-precedence"

    unchanged, selected, status = shield_visited_losses(
        legal,
        weights,
        stats,
        legal,
        selected_action=first,
    )
    assert unchanged == weights
    assert selected == first
    assert status == "all-legal-actions-certified-loss-no-safe-action"

    # All moves except the first two are UNKNOWN/unvisited; stable tie breaks by prior then UCI.
    policy, selected, status = shield_visited_losses(
        legal,
        weights,
        stats,
        (first,),
        selected_action=first,
    )
    assert selected == second
    assert status == "visited-losses-epsilon-shielded"
    assert len(policy) == len(legal) and all(p > 0 for p in policy)
    assert math.fsum(policy) == pytest.approx(1.0, abs=1e-15)


def test_visited_loss_certificate_rejects_illegal_candidate_and_keeps_rng_free_search_budget():
    rules = PythonChessRules()
    state = ChessState(chess.STARTING_FEN)
    with pytest.raises(ValueError, match="legal at the root"):
        visited_mate_in_one_losses(rules, state, (ChessMove("a1a8"),))

    evaluator = FixedEvaluator(rules, {})
    rng = random.Random(29)
    before = rng.getstate()
    result = WavefrontGumbel(
        evaluator, rules, FullGumbelConfig(16, 4, 0.0, 0.1, 50.0, True)
    ).search_many([state], [rng])[0]
    after_search = rng.getstate()
    visited = tuple(row.move for row in result.moves if row.visits > 0)
    proofs = visited_mate_in_one_losses(rules, state, visited)
    shield_visited_losses(
        tuple(move for move, _ in result.action_weights),
        tuple(p for _, p in result.action_weights),
        result.moves,
        tuple(move for move, _ in proofs),
        selected_action=result.selected_action,
        certified_wins=result.certified_mates,
    )
    assert before != after_search
    assert rng.getstate() == after_search
    assert result.simulations == 16
    assert sum(row.visits for row in result.moves) == 16
