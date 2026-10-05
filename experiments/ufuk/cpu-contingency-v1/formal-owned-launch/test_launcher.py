import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

P = Path(__file__).parent / "launch_cpu_formal.py"
spec = importlib.util.spec_from_file_location("launch", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_clock_original_and_absolute_ceilings():
    first = 1791173000
    assert m.clock(first, first) == (first + 4500, first + 4800)
    with pytest.raises(AssertionError):
        m.clock(first, first - 1)
    with pytest.raises(AssertionError):
        m.clock(first, m.ADMISSION)


def test_real_arg_wiring_pause_and_fresh_resume_same_deadline():
    cfg = {"python": "python"}
    row = {
        "run": "/r/run",
        "inputs": {
            k: {"path": "/" + k}
            for k in ("initial_weights", "book", "experiment_config", "protocol")
        },
    }
    one = m.producer_command(cfg, row, 123, False)
    two = m.producer_command(cfg, row, 123, True)
    assert one[one.index("--max-epochs") + 1] == "8"
    assert one[one.index("--stop-at") + 1] == "1"
    assert (
        "--stop-at" not in two
        and two[two.index("--resume") + 1] == "/r/run/checkpoints/epoch-00000001"
    )
    assert one[one.index("--deadline-epoch") + 1] == two[two.index("--deadline-epoch") + 1] == "123"


def test_recycled_pid_never_signalled(monkeypatch):
    monkeypatch.setattr(m, "ticks", lambda pid: (99, "S"))
    child = SimpleNamespace(pid=123, poll=lambda: None)
    calls = []
    monkeypatch.setattr(m.os, "pidfd_open", lambda pid: calls.append(pid))
    with pytest.raises(RuntimeError, match="PID reused"):
        m.stop_owned(child, 12, {123: 12})
    assert calls == []


def test_atomic_publish_never_overwrites(tmp_path):
    p = tmp_path / "receipt.json"
    m.publish(p, {"old": 1})
    with pytest.raises(FileExistsError):
        m.publish(p, {"new": 2})
    assert m.read(p) == {"old": 1}


def test_initial_admission_guard_precedes_actual_Popen():
    import ast

    tree = ast.parse(P.read_text())
    launch = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "launch"
    )
    calls = [
        n
        for n in ast.walk(launch)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen"
    ]
    asserts = [
        n
        for n in ast.walk(launch)
        if isinstance(n, ast.Assert) and "ADMISSION" in ast.unparse(n.test)
    ]
    assert (
        len(calls) == 1 and len(asserts) == 2 and max(n.lineno for n in asserts) < calls[0].lineno
    )
