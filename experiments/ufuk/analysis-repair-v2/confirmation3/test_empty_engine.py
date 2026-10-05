import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
REPO = next((p for p in ROOT.resolve().parents if (p / "pyproject.toml").is_file()), Path.cwd())
ORIGINAL = REPO / "experiments/ufuk/confirmation3/a100-mc-strength-analysis.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NEW = load(ROOT / "a100-mc-strength-analysis-v2.py", "new_analysis")
OLD = load(
    ORIGINAL,
    "old_analysis",
)
ENTRY = {"book": str(ROOT / "fixtures/book.json")}
ARM = json.loads((ROOT / "fixtures/two-real-games.json").read_text())["arm"]
ENTRY["final_sha256"] = ARM["candidate_sha256"]
BOOK = json.loads(Path(ENTRY["book"]).read_text())
EMPTY = next(g for g in ARM["games"] if g["stockfish_nodes_by_move"] == [])
NONEMPTY = next(g for g in ARM["games"] if g["stockfish_nodes_by_move"])


def check_game(game):
    result = {**ARM, "games": [game]}
    NEW.validate_arm(result, ENTRY["book"], ENTRY["final_sha256"])
    NEW.audit_trajectories(result, BOOK, stockfish=True)


def test_actual_immediate_terminal_zero_opponent_moves():
    assert EMPTY["plies"] == len(EMPTY["opening"]) + 1
    check_game(EMPTY)
    with pytest.raises(ValueError, match="node receipts missing"):
        OLD.validate_arm({**ARM, "games": [EMPTY]}, ENTRY["book"], ENTRY["final_sha256"])


def test_missing_real_opponent_receipts_rejected():
    changed = copy.deepcopy(NONEMPTY)
    changed["stockfish_nodes_by_move"] = []
    with pytest.raises(ValueError, match="node receipts missing"):
        check_game(changed)


@pytest.mark.parametrize(
    "bad", [None, {}, [{"ply": 999, "nodes": 512}], [{"ply": 17, "nodes": -1}]]
)
def test_empty_list_not_missing_or_spurious_receipts(bad):
    changed = copy.deepcopy(EMPTY)
    changed["stockfish_nodes_by_move"] = bad
    with pytest.raises(ValueError, match="node receipts missing"):
        check_game(changed)


def test_full_trajectory_still_rejects_wrong_outcome():
    changed = copy.deepcopy(EMPTY)
    changed["score"] = 0 if changed["score"] != 0 else 1
    with pytest.raises(ValueError, match="terminal outcome/perspective mismatch"):
        check_game(changed)


def test_nonzero_opponent_exact_ply_still_required():
    changed = copy.deepcopy(NONEMPTY)
    changed["stockfish_nodes_by_move"][0]["ply"] += 1
    with pytest.raises(ValueError, match="node receipts missing"):
        check_game(changed)


def test_all_statistical_functions_and_frozen_constants_unchanged():
    import ast

    def parts(path):
        tree = ast.parse(Path(path).read_text())
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
        }

    old = parts(ORIGINAL)
    new = parts(ROOT / "a100-mc-strength-analysis-v2.py")
    assert {k: v for k, v in old.items() if k != "validate_arm"} == {
        k: v for k, v in new.items() if k != "validate_arm"
    }
    for name in ("SEEDS", "REPLICATES", "CONFIDENCE", "ANALYSIS_SEED", "BOOKS"):
        assert getattr(OLD, name) == getattr(NEW, name)
