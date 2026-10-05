import ast
import hashlib
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
ORIGINAL = HERE.parent / "frozen-helper-v1"
if not ORIGINAL.is_dir():
    ORIGINAL = Path(
        "/workspace/HarbiChess/experiments/ufuk/own-terminal-visited-loss/frozen-helper-v1"
    )


def test_repair_exactly_four_metadata_literals_and_no_other_factory_logic():
    original = (ORIGINAL / "own8_formal_config_factory.py").read_text()
    expected = (
        original.replace('"qualified_ancestor_core_sha256":', '"failed_ancestor_core_sha256":')
        .replace('"qualified_ancestor_guard_sha256":', '"failed_ancestor_guard_sha256":')
        .replace(
            'a.qualified_ancestor_helpers / "own6_audit_core.py"',
            'a.qualified_ancestor_helpers / "own7_audit_core.py"',
        )
        .replace(
            'a.qualified_ancestor_helpers / "own6_adapter_controls.py"',
            'a.qualified_ancestor_helpers / "own7_adapter_controls.py"',
        )
    )
    assert (HERE / "own8_formal_config_factory.py").read_text() == expected


def expressions(path):
    tree = ast.parse(path.read_text())
    pair = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in (
                    "failed_ancestor_core_sha256",
                    "failed_ancestor_guard_sha256",
                    "qualified_ancestor_core_sha256",
                    "qualified_ancestor_guard_sha256",
                    "ancestor_profile_qualified",
                ):
                    pair[key.value] = value
    return pair


def test_actual_new_metadata_expressions_read_real_old7_bytes_old6_absence_fails(tmp_path):
    old = expressions(ORIGINAL / "own8_formal_config_factory.py")
    new = expressions(HERE / "own8_formal_config_factory.py")

    class Arguments:
        qualified_ancestor_helpers = tmp_path

    scope = {
        "a": Arguments(),
        "sha": lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
        "ancestor_profile_qualified": False,
    }
    for kind in ("core", "guard"):
        filename = "own7_audit_core.py" if kind == "core" else "own7_adapter_controls.py"
        data = ("actual-preserved-source7-" + kind).encode()
        (tmp_path / filename).write_bytes(data)
        expression = ast.Expression(new["failed_ancestor_" + kind + "_sha256"])
        assert (
            eval(compile(expression, "actual-new-factory-metadata", "eval"), scope)
            == hashlib.sha256(data).hexdigest()
        )
        with pytest.raises(FileNotFoundError):
            eval(
                compile(
                    ast.Expression(old["qualified_ancestor_" + kind + "_sha256"]),
                    "original-stale-factory-metadata",
                    "eval",
                ),
                scope,
            )
    assert (
        eval(
            compile(
                ast.Expression(new["ancestor_profile_qualified"]), "actual-ancestry-status", "eval"
            ),
            scope,
        )
        is False
    )


def test_template_changes_only_admission_explanation_and_unregistered_reserve():
    original = json.loads((ORIGINAL / "protocol-DRAFT.json").read_text())
    new = json.loads((HERE / "protocol-DRAFT.json").read_text())
    assert new["posttraining_reserve_seconds"] == 6000
    for key in set(original) | set(new):
        if key not in ("compute_admission", "posttraining_reserve_seconds"):
            assert new[key] == original[key], key
    assert "CUDA95" in new["compute_admission"] and "03:10" in new["compute_admission"]
    assert "04:15" in new["compute_admission"] and "04:40" in new["compute_admission"]
    assert new["configs"] == original["configs"] and new["strength"] == original["strength"]
