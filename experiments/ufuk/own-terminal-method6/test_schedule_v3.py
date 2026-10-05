import ast
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("schedule6", HERE / "own6_schedule_v3.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def reg():
    return {
        "scheduling_version": "prospective-own6-terminal45-scheduling-v3",
        "absolute_audit_cutoff_epoch": s.AUDIT_CUTOFF,
        "latest_latency_start_epoch": s.LATEST_LATENCY_START,
        "whole_training_seconds_per_seed": 6000,
        "whole_audit_seconds_from_originalfirstclock": 9000,
        "earliest_training_epoch": 1791163000,
    }


def test_audit9000_maximum_and_actual0325_absolute_cutoff():
    assert s.audit_deadline(1791163000, 9000) == s.AUDIT_CUTOFF
    assert s.audit_deadline(1791160000, 9000) == 1791169000
    with pytest.raises(AssertionError):
        s.audit_deadline(1791163000, 12600)


def test_original_training6000_clock_no_future_reset_cutoff_failclosed():
    r = reg()
    first = 1791163000
    s.validate_clock(r, first, first + 6000, first + 1)
    for badfirst, badnow in [(first + 100, first + 1), (first, s.AUDIT_CUTOFF)]:
        with pytest.raises(AssertionError):
            s.validate_clock(r, badfirst, badfirst + 6000, badnow)
    with pytest.raises(AssertionError):
        s.validate_clock(r, first, first + 6001, first + 1)


def test_physical06_caps_phase_without_changing_max7300():
    assert s.phase_deadline(1791165000, 7300) == 1791172300
    assert s.phase_deadline(s.LATEST_LATENCY_START + 180, 7300) == s.END - 1
    with pytest.raises(AssertionError):
        s.phase_deadline(s.END, 7300)


def test_no_fabricated_old_latency_or_final_wait_in_new_barrier():
    tree = ast.parse((HERE / "own6_previous_methods_barrier.py").read_text())
    text = ast.unparse(tree)
    assert "verify_previous" in text
    assert "cohort-latency-complete" not in text and "final-" not in text
    guardian = ast.unparse(ast.parse((HERE / "own6_protect_original45.py").read_text()))
    assert "kill" not in guardian and "send_signal" not in guardian


def test_inner_final_and_replay_deadlines_use_same_hardcap_function():
    tree = ast.parse((HERE / "own6_posttraining.py").read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = [
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "phase_deadline"
    ]
    assert any(
        len(n.args) == 2
        and isinstance(n.args[1], ast.Constant)
        and n.args[1].value == 600
        for n in calls
    )
    assert "bind_argument_deadline(arguments, deadline)" in ast.unparse(main)
    deadline = s.phase_deadline(s.LATEST_LATENCY_START + 180, 7300)
    assert s.bind_argument_deadline(["--deadline-epoch", s.END], deadline) == [
        "--deadline-epoch",
        str(deadline),
    ]
    assert (
        float(s.bind_argument_deadline(["--deadline-epoch", deadline - 5], deadline)[1])
        == deadline - 5
    )
    assert "prior45-incomplete-owned-termination-only.json" in ast.unparse(main)


def test_fixed_hyperparams_source_and_epochs_kept_in_factory():
    tree = ast.parse((HERE / "own6_formal_config_factory.py").read_text())
    text = ast.unparse(tree)
    assert "assert epochs == 8" in text
    assert "a.whole_training_seconds == 6000" in text
    assert "a.whole_audit_seconds == 9000" in text
    assert "check_unit34_receipt" in text and "check_cli_receipt" in text
    assert "terminal45_binding" in text


def test_actual_proc_snapshot_has_group_keys_needed_by_terminal45_barrier():
    spec = importlib.util.spec_from_file_location(
        "terminal_release6", HERE / "own6_terminal45_release.py"
    )
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    table = release.process_table()
    assert table and all("pgid" in row for row in table.values())


def test_factory_admits_binding_repair_only_with_same_original900_clock():
    tree = ast.parse((HERE / "own6_formal_config_factory.py").read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "check_profile_clock"
    )
    scope = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "factory-clock", "exec"),
        scope,
    )
    check = scope["check_profile_clock"]
    first = 1791163947.632503
    profile = {
        "schema": "owned900-search-acting-v2-fullshape-development-v2-originalclock-bindingrepair",
        "original_started_epoch": first,
        "started_epoch": first + 200,
        "finished_epoch": first + 899,
        "absolute_deadline_epoch": first + 900,
    }
    check(profile)
    for patch in (
        {"absolute_deadline_epoch": first + 1100},
        {"original_started_epoch": first + 200},
        {"finished_epoch": first + 901},
        {"started_epoch": first - 1},
    ):
        with pytest.raises(AssertionError):
            check({**profile, **patch})
