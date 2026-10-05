import ast
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("owned", HERE / "launch_formal7_owned.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_cross_group_owned_children_and_pid_reuse_protect_foreign_processes():
    roots = {10: dict(pid=10, startticks=1)}
    table = {
        10: dict(pid=10, startticks=1, ppid=1, pgid=10),
        11: dict(pid=11, startticks=2, ppid=10, pgid=11),
        12: dict(pid=12, startticks=3, ppid=11, pgid=12),
        20: dict(pid=20, startticks=4, ppid=1, pgid=20),
    }
    assert set(m.descendants(roots, table)) == {10, 11, 12}
    table[10]["startticks"] = 99
    assert m.descendants(roots, table) == {}


def test_actual_commonclock_future_and_admission_mutations_fail():
    m.guard(m.ADMISSION - 5, m.ADMISSION - 4)
    with pytest.raises(AssertionError):
        m.guard(m.ADMISSION - 5, m.ADMISSION - 6)
    with pytest.raises(AssertionError):
        m.guard(m.ADMISSION + 1, m.ADMISSION + 2)
    with pytest.raises(AssertionError):
        m.guard(m.ADMISSION - 5, m.END)


def test_all_actual_launch_calls_bind_nested_function_signature():
    tree = ast.parse((HERE / "launch_formal7_owned.py").read_text())
    launch = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "launch"
    )
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "launch"
        ):
            assert len(call.args) == len(launch.args.args) == 2
    assert "signal.pidfd_send_signal" in ast.unparse(tree)
    assert "os.killpg" not in ast.unparse(tree)


def test_metadata_publication_never_overwrites_clock_or_failure(tmp_path):
    path = tmp_path / "common-original-firstclock.json"
    m.publish(path, {"clock": 123})
    with pytest.raises(FileExistsError):
        m.publish(path, {"clock": 456})
    assert "123" in path.read_text()
