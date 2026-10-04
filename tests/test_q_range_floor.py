import random

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS, _Node
from harbichess.search.q_range_floor import QRangeFlooredMCTS


class UniformEvaluator:
    def __init__(self, rules):
        self.rules = rules

    def evaluate(self, state):
        moves = self.rules.legal_moves(state)
        return PositionEvaluation(tuple((move, 1 / len(moves)) for move in moves), 0.1)


def test_default_is_exact_and_floor_preserves_real_mate_backup_and_budget():
    rules = PythonChessRules()
    evaluator = UniformEvaluator(rules)
    config = FullGumbelConfig(simulations=32, max_considered_actions=16)
    old = FullGumbelMCTS(evaluator, rules=rules, config=config)
    default = QRangeFlooredMCTS(evaluator, rules=rules, config=config)
    for state in (
        rules.initial_state(),
        rules.initial_state("8/8/8/8/8/8/8/k1KQ4 w - - 0 1"),
    ):
        assert old.search(state, rng=random.Random(5)) == default.search(
            state, rng=random.Random(5)
        )
    floor = QRangeFlooredMCTS(evaluator, rules=rules, config=config, range_floor=0.2)
    result = floor.search(
        rules.initial_state("8/8/8/8/8/8/8/k1KQ4 w - - 0 1"), rng=random.Random(5)
    )
    assert result.selected_action.uci == "d1a4"
    mate = next(move for move in result.moves if move.move.uci == "d1a4")
    assert mate.mean_value == pytest.approx(1)
    assert sum(move.visits for move in result.moves) == 32
    assert sum(weight for _, weight in result.action_weights) == pytest.approx(1)


def test_small_range_damps_confidence_keeps_rank_and_large_range_unchanged():
    rules = PythonChessRules()
    old = FullGumbelMCTS(UniformEvaluator(rules))
    floor = QRangeFlooredMCTS(UniformEvaluator(rules), range_floor=0.2)
    node = _Node(
        children={
            ChessMove("e2e4"): _Node(prior=0.4, visit_count=2, value_sum=-0.02),
            ChessMove("d2d4"): _Node(prior=0.4, visit_count=2, value_sum=0.02),
            ChessMove("g1f3"): _Node(prior=0.2),
        }
    )
    before, after = old._completed_q(node), floor._completed_q(node)
    assert sorted(before, key=before.get) == sorted(after, key=after.get)
    assert max(after.values()) - min(after.values()) < max(before.values()) - min(before.values())
    for child in node.children.values():
        child.value_sum *= 100
    assert old._completed_q(node) == floor._completed_q(node)
    for invalid in (True, float("inf"), float("nan"), 0, -1):
        with pytest.raises(ValueError, match="Q range floor"):
            QRangeFlooredMCTS(UniformEvaluator(rules), range_floor=invalid)
