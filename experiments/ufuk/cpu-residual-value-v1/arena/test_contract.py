"""Pure controls: role remapping preserves entire audit logic and search bytes."""

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = Path("/workspace/HarbiChess/experiments/ufuk/cpu-shrunk-value-v1/arena")


def test_gate_function_ast_and_search_exact_legacy_bytes():
    original = ast.parse((OLD / "audit_arena.py").read_text())
    changed = ast.parse((HERE / "audit_arena.py").read_text().replace("residual", "shrunk"))
    assert ast.dump(original, include_attributes=False) == ast.dump(
        changed, include_attributes=False
    )
    assert (HERE / "search.py").read_bytes() == (OLD / "search.py").read_bytes()
    assert hashlib.sha256((HERE / "search.py").read_bytes()).hexdigest() == (
        "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
    )


def test_template_fixed_roles_gates_and_zero_control_duplication():
    old = json.loads((OLD / "protocol.json").read_text())
    new = json.loads((HERE / "protocol-TEMPLATE.json").read_text())
    assert new["match_seeds"] == [20262705, 20262706]
    assert new["tasks"] == [
        ["e8", "SF512"],
        ["rebased", "SF512"],
        ["residual", "e8"],
        ["residual", "rebased"],
        ["residual", "SF512"],
    ]
    assert new["screen"] == old["screen"]
    for key in (
        "total_games",
        "opening_pairs",
        "games_per_tournament",
        "search_nodes",
        "quiescence_plies",
        "max_depth",
        "max_plies",
        "stockfish_nodes",
    ):
        assert new[key] == old[key]
    assert "not independent families" in new["untrained_control"]
