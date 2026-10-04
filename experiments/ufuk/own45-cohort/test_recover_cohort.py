import ast
import importlib.util
from pathlib import Path

import pytest

FILE = Path(__file__).with_name("recover_cohort.py")
spec = importlib.util.spec_from_file_location("recovery", FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
FIRST = 1791156456.0081534


def clock(first=FIRST):
    return {
        "schema": "own45-common-original-firstclock-v1",
        "slots": [4, 5],
        "original_training_started_epoch": first,
    }


def test_original_clock_not_replaced_by_present():
    assert m.validate_clock(clock(), FIRST, FIRST + 600) == FIRST
    with pytest.raises(AssertionError):
        m.validate_clock(clock(FIRST + 600), FIRST, FIRST + 601)


def test_future_stale_and_changed_clock_fail():
    for recorded, expected, now in (
        (FIRST + 1, FIRST + 1, FIRST),
        (m.END - 22619, m.END - 22619, m.END - 22000),
        (FIRST, FIRST + 1, FIRST + 600),
    ):
        with pytest.raises(AssertionError):
            m.validate_clock(clock(recorded), expected, now)


def test_changed_bytes_rejected(tmp_path):
    p = tmp_path / "clock.json"
    p.write_text("original clock")
    expected = m.sha(p)
    p.write_text("replacement clock")
    assert m.sha(p) != expected


def test_actual_nested_launch_wiring_and_no_new_clock():
    tree = ast.parse(FILE.read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    launch = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.FunctionDef) and n.name == "launch"
    )
    signature_count = len(launch.args.args)
    calls = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "launch"
    ]
    assert len(calls) == 5  # owner config, then 4 kinds inside fixed loops
    assert all(len(n.args) == signature_count and not n.keywords for n in calls)
    assigns = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in ("first", "baseline_first")
            for t in n.targets
        )
    ]
    assert all("time.time" not in ast.unparse(n.value) for n in assigns)
    text = FILE.read_text()
    assert 'recovery["owner_factories"][str(slot)]["path"]' in text
    assert '"--output-sha256"' not in text
    assert 'registration = old_stage / ("registration-" + str(slot))' in text
    assert "helper(stage, method," not in ast.unparse(main)


def test_actual_helper_uses_old_stage():
    assert m.helper(
        Path("/old"), {"slot": 4, "helper_subdir": "helpers4"}, "baseline"
    ) == Path("/old/helpers4/ownv1_baseline.py")
