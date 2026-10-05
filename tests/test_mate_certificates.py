import random

import chess
import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.full_gumbel import FullGumbelConfig
from harbichess.search.mate_certificates import certified_policy, immediate_mating_moves
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
