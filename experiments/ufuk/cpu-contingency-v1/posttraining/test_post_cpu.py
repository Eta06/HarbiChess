import ast
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

P = Path(__file__).parent / "post_cpu.py"
spec = importlib.util.spec_from_file_location("post_cpu", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_historical_completed_receipt_read_before_deadline(tmp_path):
    p = tmp_path / "done.json"
    p.write_text(json.dumps({"status": "completed"}))
    assert m.wait_json(p, 1)["status"] == "completed"
    p.write_text("{")
    with pytest.raises(TimeoutError):
        m.wait_json(p, 1)


def test_failed_semantic_status_is_not_hidden_or_retried(tmp_path):
    p = tmp_path / "failed.json"
    p.write_text(json.dumps({"status": "failed"}))
    assert m.wait_json(p, 1)["status"] == "failed"


def test_all_actual_run_calls_bind_same_signature():
    tree = ast.parse(P.read_text())
    signature = inspect.Signature(
        [
            inspect.Parameter(
                n,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=False if n == "qualified" else inspect.Parameter.empty,
            )
            for n in ("name", "helper", "arguments", "ceiling", "qualified")
        ]
    )
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "run"
    ]
    assert len(calls) == 6
    for call in calls:
        signature.bind(*([object()] * len(call.args)), **{kw.arg: object() for kw in call.keywords})
    eligibility = next(
        c for c in calls if isinstance(c.args[0], ast.Constant) and c.args[0].value == "eligibility"
    )
    assert (
        eligibility.keywords[0].arg == "qualified" and eligibility.keywords[0].value.value is False
    )


def test_hardend_and_CPU_contaminants_exact():
    assert m.END == 1791180000
    for command in (
        "python -m harbichess.evaluation.portable_arena",
        "python cpu_contingency_full_audit.py",
        "python -m pytest",
        "python cpu_contingency_parity.py",
    ):
        assert m.BUSY.search(command)
    assert not m.BUSY.search("python post_cpu.py --config /tmp/post.json --execute")
