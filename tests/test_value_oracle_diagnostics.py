from dataclasses import dataclass

from harbichess.chess.rules import PythonChessRules
from harbichess.evaluation.value_oracle_diagnostics import _search_rows
from harbichess.search.evaluator import PositionEvaluation


@dataclass(frozen=True)
class Record:
    state: object


class UniformEvaluator:
    def __init__(self, rules: PythonChessRules) -> None:
        self.rules = rules

    def evaluate(self, state) -> PositionEvaluation:
        moves = self.rules.legal_moves(state)
        return PositionEvaluation(tuple((move, 1.0 / len(moves)) for move in moves), 0.0)


def test_value_oracle_diagnostic_supports_forced_move_positions() -> None:
    rules = PythonChessRules()
    state = rules.initial_state("5k2/8/8/5p2/2p1p3/8/7q/5K2 w - - 8 71")

    rows = _search_rows(
        UniformEvaluator(rules),
        rules=rules,
        records=(Record(state),),
        budget=8,
        workers=1,
        seed=1,
        arm="test",
    )

    assert rows[0]["legal_children"] == 1
    assert rows[0]["selected_move"].uci == "f1e1"
    assert rows[0]["visit_margin"] == 8
