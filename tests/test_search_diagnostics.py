from pathlib import Path
from typing import ClassVar

import chess
import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.evaluation.search_diagnostics import (
    SearchDiagnosticConfig,
    audit_search_conventions,
)
from harbichess.search.diagnostics import (
    TACTICAL_CASES,
    TacticalCase,
    run_tactical_sweep,
    validate_tactical_cases,
)
from harbichess.search.evaluator import PositionEvaluation


class MaterialEvaluator:
    values: ClassVar[dict[int, int]] = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def __init__(self, rules: PythonChessRules) -> None:
        self.rules = rules

    def evaluate(self, state) -> PositionEvaluation:
        board = self.rules.inspect(state)
        own = sum(
            len(board.pieces(piece, board.turn)) * value
            for piece, value in self.values.items()
        )
        opponent = sum(
            len(board.pieces(piece, not board.turn)) * value
            for piece, value in self.values.items()
        )
        return PositionEvaluation(
            tuple((move, 1.0) for move in self.rules.legal_moves(state)),
            max(-1.0, min(1.0, (own - opponent) / 15)),
        )


def test_tactical_fixtures_are_proven_by_rules() -> None:
    validate_tactical_cases()

    assert {case.category for case in TACTICAL_CASES} == {
        "mate-in-one",
        "mate-in-two",
        "forced-defense",
        "hanging-piece",
    }


def test_search_conventions_preserve_sign_and_history() -> None:
    audit = audit_search_conventions()

    assert audit["passed"] is True
    assert audit["terminal_value"] == -1
    assert audit["backed_up_values_root_to_leaf"] == (1.0, -1.0, 1.0, -1.0)
    assert all(audit["checks"].values())


def test_tactical_sweep_reports_budget_regressions_and_oracle_mass() -> None:
    rules = PythonChessRules()
    result = run_tactical_sweep(
        MaterialEvaluator(rules),
        rules=rules,
        budgets=(8, 32),
        workers=2,
        seed=17,
        cases=(TACTICAL_CASES[0], TACTICAL_CASES[-2]),
    )

    assert result["raw"]["total"] == 2
    assert [sweep["budget"] for sweep in result["budgets"]] == [8, 32]
    assert all(
        row["expected_policy_mass"] > 0 for row in result["raw"]["cases"]
    )
    assert all(
        row["visited_children"] <= row["legal_children"]
        for sweep in result["budgets"]
        for row in sweep["cases"]
    )


def test_tactical_sweep_supports_full_gumbel_allocation() -> None:
    rules = PythonChessRules()
    result = run_tactical_sweep(
        MaterialEvaluator(rules),
        rules=rules,
        budgets=(8, 16),
        workers=2,
        seed=17,
        search_kind="full-gumbel",
        max_considered_actions=4,
        cases=(TACTICAL_CASES[0], TACTICAL_CASES[-2]),
    )

    assert [sweep["budget"] for sweep in result["budgets"]] == [8, 16]
    assert all(
        sum(row["leader_visits"] for row in sweep["cases"]) > 0
        for sweep in result["budgets"]
    )


def test_tactical_sweep_rejects_bad_schedule_and_fixture() -> None:
    rules = PythonChessRules()
    evaluator = MaterialEvaluator(rules)
    with pytest.raises(ValueError, match="increasing"):
        run_tactical_sweep(
            evaluator,
            rules=rules,
            budgets=(32, 8),
            workers=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="invalid tactical oracle"):
        validate_tactical_cases(
            (
                TacticalCase(
                    "bad-mate",
                    "mate-in-one",
                    TACTICAL_CASES[0].fen,
                    ("d1d2",),
                ),
            )
        )
    with pytest.raises(ValueError, match="configuration"):
        SearchDiagnosticConfig(
            run_result=Path("run.json"),
            shard=Path("replay.jsonl.gz"),
            output_dir=Path("diagnostics"),
            budgets=(32, 8),
        )
