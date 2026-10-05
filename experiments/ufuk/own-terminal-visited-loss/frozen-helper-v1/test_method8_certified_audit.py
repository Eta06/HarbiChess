import copy
import os
import random
import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parent))
if os.environ.get("METHOD8_SOURCE"):
    sys.path.insert(0, os.environ["METHOD8_SOURCE"])


from own8_audit_core import audit_winning_root_receipt, replay_mate_certificate_inventory

from harbichess.chess.actions import legal_action_indices, move_to_action
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS
from harbichess.search.ownsearch_wavefront import WavefrontGumbel


class Evaluator:
    def __init__(self, rules):
        self.rules = rules

    def evaluate_many(self, states):
        rows = []
        for state in states:
            moves = self.rules.legal_moves(state)
            probs = [1 / len(moves)] * len(moves)
            rows.append(PositionEvaluation(tuple(zip(moves, probs, strict=True)), 0.0))
        return tuple(rows)

    def evaluate(self, state):
        return self.evaluate_many([state])[0]


def packet(rules, state):
    legal_actions = tuple(legal_action_indices(rules.inspect(state)))
    result = WavefrontGumbel(
        Evaluator(rules),
        rules,
        FullGumbelConfig(16, 4, 0.0, 0.1, 50.0, True),
    ).search_many([state], [random.Random(23)])[0]
    weights = {
        move_to_action(rules.inspect(state), chess.Move.from_uci(move.uci)): value
        for move, value in result.action_weights
    }
    return legal_actions, {
        "certified_mates": [move.uci for move in result.certified_mates],
        "moves": [
            {
                "move": item.move.uci,
                "visits": item.visits,
                "prior": item.prior,
                "mean_value": item.mean_value,
            }
            for item in result.moves
        ],
        "simulations": result.simulations,
        "selected_action": result.selected_action.uci,
        "root_value": result.root_value,
        "search_policy": [weights[action] for action in legal_actions],
    }


def test_independent_fullhistory_checker_reports_all_winning_moves_and_perspective():
    rules = PythonChessRules()
    fixtures = [
        (
            ChessState("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"),
            {"f7f8", "f7e8", "f7h7", "f7g7"},
        ),
        (
            ChessState("7K/8/7k/1q6/8/8/8/8 b - - 0 1"),
            {"b5e8", "b5b8"},
        ),
    ]
    for state, expected in fixtures:
        listed = sorted(expected)
        assert (
            set(replay_mate_certificate_inventory(state, listed, rules, claim_draw=True))
            == expected
        )
        with pytest.raises(AssertionError):
            replay_mate_certificate_inventory(state, listed[:-1], rules, claim_draw=True)
        with pytest.raises(AssertionError):
            replay_mate_certificate_inventory(state, [*listed, "a1a8"], rules, claim_draw=True)


def test_claimable_draw_root_has_empty_certificate_inventory():
    rules = PythonChessRules()
    state = ChessState(
        chess.STARTING_FEN,
        tuple(ChessMove(move) for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2),
    )
    assert rules.outcome(state, claim_draw=True) is not None
    assert replay_mate_certificate_inventory(state, [], rules, claim_draw=True) == ()
    with pytest.raises(AssertionError):
        replay_mate_certificate_inventory(state, ["e2e4"], rules, claim_draw=True)


def test_both_search_engines_honor_claim_draw_for_full_history():
    rules = PythonChessRules()
    state = ChessState(
        chess.STARTING_FEN,
        tuple(ChessMove(move) for move in ("g1f3", "g8f6", "f3g1", "f6g8") * 2),
    )
    results = {}
    for claim_draw in (True, False):
        config = FullGumbelConfig(16, 4, 0.0, 0.1, 50.0, claim_draw)
        rng1 = random.Random(101)
        rng2 = random.Random(101)
        results[(claim_draw, "full")] = FullGumbelMCTS(
            Evaluator(rules), rules=rules, config=config
        ).search(state, rng=rng1)
        results[(claim_draw, "wavefront")] = WavefrontGumbel(
            Evaluator(rules), rules, config
        ).search_many([state], [rng2])[0]

    for engine in ("full", "wavefront"):
        claimed = results[(True, engine)]
        unclaimed = results[(False, engine)]
        assert claimed.simulations == 0
        assert claimed.outcome is not None
        assert unclaimed.simulations == 16
        assert unclaimed.outcome is None


def test_corrupt_certified_root_receipt_mutations_are_rejected():
    rules = PythonChessRules()
    state = ChessState("7K/8/7k/1q6/8/8/8/8 b - - 0 1")
    legal, exact = packet(rules, state)
    assert exact["certified_mates"] == ["b5b8", "b5e8"]
    assert (
        audit_winning_root_receipt(state, exact, legal, rules, claim_draw=True, simulations=16) == 2
    )

    mutations = {}
    mutations["missing certificate"] = copy.deepcopy(exact)
    mutations["missing certificate"]["certified_mates"] = ["b5b8"]
    mutations["extra nonmate"] = copy.deepcopy(exact)
    mutations["extra nonmate"]["certified_mates"].append("b5a5")
    mutations["wrong selected action"] = copy.deepcopy(exact)
    mutations["wrong selected action"]["selected_action"] = "b5a5"
    mutations["wrong mover perspective"] = copy.deepcopy(exact)
    mutations["wrong mover perspective"]["root_value"] = -1.0
    mutations["visit budget"] = copy.deepcopy(exact)
    mutations["visit budget"]["moves"][-1]["visits"] -= 1
    mutations["policy support"] = copy.deepcopy(exact)
    selected_index = legal.index(
        move_to_action(rules.inspect(state), chess.Move.from_uci(exact["selected_action"]))
    )
    mutations["policy support"]["search_policy"][selected_index] = 0.0
    mutations["nonmate target mass"] = copy.deepcopy(exact)
    nonmate_action = next(
        action
        for action in legal
        if all(
            move_to_action(rules.inspect(state), chess.Move.from_uci(move)) != action
            for move in exact["certified_mates"]
        )
    )
    nonmate_index = legal.index(nonmate_action)
    mutations["nonmate target mass"]["search_policy"][nonmate_index] += 0.1
    mutations["schema-style duplicate"] = copy.deepcopy(exact)
    mutations["schema-style duplicate"]["certified_mates"].append("b5e8")

    for name, corrupt in mutations.items():
        if name in {"missing certificate", "extra nonmate", "schema-style duplicate"}:
            with pytest.raises(AssertionError):
                replay_mate_certificate_inventory(
                    state, corrupt["certified_mates"], rules, claim_draw=True
                )
        else:
            with pytest.raises(AssertionError):
                audit_winning_root_receipt(
                    state, corrupt, legal, rules, claim_draw=True, simulations=16
                )
