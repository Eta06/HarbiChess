import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
REPO = next((p for p in ROOT.resolve().parents if (p / "pyproject.toml").is_file()), Path.cwd())


def binder():
    spec = importlib.util.spec_from_file_location("binding3", ROOT / "analysis_repair_binding.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_exact_primary_count_metadata_exception_and_all_other_asserts():
    m = binder()
    for suffix in ("strength_analysis", "all_gates"):
        m.unchanged_checks(
            REPO / f"experiments/ufuk/search-acting-method5/own5_{suffix}.py",
            ROOT / f"own5_{suffix}.py",
            suffix == "strength_analysis",
        )
    assert '"primary_comparisons": 4' in (ROOT / "own5_strength_analysis.py").read_text()
    assert 'strength["primary_comparisons"] == 4' in (ROOT / "own5_all_gates.py").read_text()


def test_any_threshold_change_still_rejected(tmp_path):
    m = binder()
    bad = tmp_path / "bad.py"
    bad.write_text((ROOT / "own5_all_gates.py").read_text().replace("> 0.6", "> 0.5"))
    with pytest.raises(AssertionError):
        m.unchanged_checks(
            REPO / "experiments/ufuk/search-acting-method5/own5_all_gates.py", bad, False
        )


def test_only_primary_metadata_constant_is_normalized():
    m = binder()
    tree = ast.parse('assert strength["primary_comparisons"] == 5\nassert gain > 5\n')
    text = ast.unparse(m.normalize_primary_count(tree))
    assert "strength['primary_comparisons'] == 4" in text
    assert "gain > 5" in text
