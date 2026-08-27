import json
from pathlib import Path

import pytest

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove
from harbichess.dashboard.state import SnapshotStore
from harbichess.evaluation.value_oracle_diagnostics import (
    ValueOracleDiagnosticConfig,
    publish_value_oracle_diagnostics,
)
from harbichess.search.evaluator import PositionEvaluation
from harbichess.search.value_oracle import (
    DeterministicTacticalOracle,
    OracleValueEvaluator,
    TacticalOracleConfig,
)


class FixedPolicyEvaluator:
    def __init__(self, rules: PythonChessRules) -> None:
        self.rules = rules

    def evaluate(self, state) -> PositionEvaluation:
        moves = self.rules.legal_moves(state)
        return PositionEvaluation(
            tuple((move, index + 1.0) for index, move in enumerate(moves)),
            -0.75,
        )


def test_oracle_preserves_policy_and_replaces_only_value() -> None:
    rules = PythonChessRules()
    state = rules.initial_state("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    policy = FixedPolicyEvaluator(rules)
    oracle = DeterministicTacticalOracle(
        rules=rules,
        config=TacticalOracleConfig(depth=2),
    )

    neural = policy.evaluate(state)
    replaced = OracleValueEvaluator(policy, oracle).evaluate(state)

    assert replaced.priors == neural.priors
    assert replaced.value == oracle.value(state)
    assert replaced.value > 0.0
    assert replaced.value != neural.value


def test_oracle_terminal_and_history_values_use_side_to_move() -> None:
    rules = PythonChessRules()
    oracle = DeterministicTacticalOracle(rules=rules)
    checkmated = rules.initial_state()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        checkmated = rules.apply(checkmated, ChessMove(move))

    assert oracle.value(checkmated) == -1.0
    assert oracle.value(checkmated) == oracle.value(checkmated)


def test_oracle_configuration_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError, match="configuration"):
        TacticalOracleConfig(depth=-1)
    with pytest.raises(ValueError, match="configuration"):
        ValueOracleDiagnosticConfig(
            run_result=Path("run.json"),
            shard=Path("replay.jsonl.gz"),
            output_dir=Path("diagnostics"),
            oracle_depth=4,
            verifier_depth=2,
        )


def test_qualified_oracle_publish_keeps_promotion_blocked(tmp_path: Path) -> None:
    result = tmp_path / "diagnostics.json"
    result.write_text(
        json.dumps(
            {
                "source_commit": "a" * 40,
                "config": {"positions": 2, "budgets": [8]},
                "gate": {
                    "bootstrap_teacher_qualified": True,
                    "qualified_oracle_budgets": [8],
                },
                "search": {
                    "oracle": {
                        "8": {
                            "mean_verified_action_value_delta": 0.2,
                            "verified_action_value_delta_95_interval": [0.1, 0.3],
                        }
                    }
                },
                "tactical": {
                    "oracle": {"aggregate_solve_count_monotonic": True, "budgets": []}
                },
            }
        ),
        encoding="utf-8",
    )
    telemetry = tmp_path / "state.json"

    publish_value_oracle_diagnostics(result, telemetry)

    snapshot = SnapshotStore(telemetry).read()
    assert snapshot.teacher_qualification_status == "passed"
    assert snapshot.teacher_qualified_variants == ("oracle-8",)
    assert not snapshot.promotion_ready
