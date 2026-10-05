import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("late_guard", HERE / "late_training_overlap_guard.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def documents():
    return tuple(
        json.loads((HERE / "fixtures" / f"{name}.json").read_text()) for name in ("main", "tiny")
    )


def test_actual_frozen_receipts_and_both_books_are_bound():
    binding = {
        name: {"path": str(HERE / "fixtures" / f"{name}.json"), "sha256": expected}
        for name, expected in [("main", m.MAIN_SHA), ("tiny", m.TINY_SHA)]
    }
    result = m.validate(binding, lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    assert result["main"] == m.MAIN_SHA and result["tiny"] == m.TINY_SHA
    bad = copy.deepcopy(binding)
    bad["tiny"]["sha256"] = m.MAIN_SHA
    with pytest.raises(AssertionError):
        m.validate(bad, lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())


@pytest.mark.parametrize(
    "which,key,value",
    [
        (0, "matched_roots_count", 1),
        (0, "candidate_roots", 48),
        (0, "journal_count", 35),
        (0, "total_closed_epoch_rows", 1179647),
        (1, "status", "unknown"),
        (1, "matched_root_keys", ["heldout-root"]),
        (1, "fullhistory_prepost_position_keys_examined", 4096),
        (1, "main_overlap_receipt_sha256", "0" * 64),
        (1, "source8_closed_epochs", [1]),
    ],
)
def test_each_missing_or_mismatched_actual_scope_is_rejected(which, key, value):
    docs = list(documents())
    docs[which][key] = value
    with pytest.raises(AssertionError):
        m.validate_documents(*docs)


def test_actual_launcher_validates_late_proof_before_any_clock_or_popen():
    text = (HERE / "launch_formal8_owned.py").read_text()
    tree = ast.parse(text)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    guard = next(
        n.lineno
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "validate"
    )
    first = next(
        n.lineno
        for n in ast.walk(main)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "first" for t in n.targets)
    )
    assert guard < first
    assert "late_training_overlap_guard_sha256" in text
