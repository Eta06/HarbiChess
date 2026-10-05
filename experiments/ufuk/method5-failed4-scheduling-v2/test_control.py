"""Pure process-identity and exact routing checks; no jobs or heldout reads."""

import ast
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
dep = importlib.import_module("failed4_dependency")
identity_live, remaining_owned = dep.identity_live, dep.remaining_owned
rewrite_argv = importlib.import_module("post5_scheduling_v2").rewrite_argv


def test_process_groups_include_reparented_children_and_ignore_zombies():
    receipt = {
        "owned_process_group_ids": [11],
        "owner_identities": [{"pid": 11, "startticks": 9}],
    }
    table = {
        11: {"pgid": 11, "startticks": 9, "state": "Z"},
        12: {"pgid": 11, "startticks": 10, "state": "S"},
        13: {"pgid": 99, "startticks": 11, "state": "S"},
    }
    assert remaining_owned(receipt, table) == [12]


def test_pid_reuse_does_not_claim_terminated_original_owner_alive():
    assert not identity_live({"pid": 42, "startticks": 1}, {42: {"startticks": 2, "state": "S"}})
    assert identity_live({"pid": 42, "startticks": 1}, {42: {"startticks": 1, "state": "S"}})


def test_only_analysis_and_gate_commands_change_with_truthful_explicit_binding():
    s = {
        "analysis_v3_directory": "/new",
        "analysis_v3_supplement": "/new/supp.json",
        "analysis_v3_supplement_sha256": "a" * 64,
        "original_helpers": "/old",
        "original_registration": "/old/reg.json",
        "analysis_v2_supplement": "/v2/supp.json",
    }
    for name in (
        "own5_fresh_cli_replay.py",
        "own5_eligibility.py",
        "own5_parity.py",
        "own5_latency.py",
        "own5_final_arms.py",
    ):
        argv = ["python", "/old/" + name, "--deadline-epoch", "123"]
        assert rewrite_argv(argv, s) == argv
    for name in ("own5_strength_analysis.py", "own5_all_gates.py"):
        argv = [
            "python",
            "/old/" + name,
            "--qualification-config",
            "/old/q.json",
            "--qualification-config-sha256",
            "b" * 64,
        ]
        result = rewrite_argv(argv, s)
        assert result[:2] == ["python", "/new/" + name]
        assert result[2 : len(argv)] == argv[2:]
        assert result[-2:] == ["--previous-analysis-repair", "/v2/supp.json"]
        assert "--original-helpers" in result


def test_original_coordinator_launch_calls_bind_actual_function_signature():
    repo = Path("/workspace/HarbiChess")
    if not repo.is_dir():
        repo = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
    module = ast.parse(
        (repo / "experiments/ufuk/search-acting-method5/own5_posttraining.py").read_text()
    )
    launch = next(
        n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == "launch"
    )
    positional = len(launch.args.args)
    required = positional - len(launch.args.defaults)
    calls = [
        n
        for n in ast.walk(module)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "launch"
    ]
    assert len(calls) == 7
    for call in calls:
        assert required <= len(call.args) <= positional
        assert {k.arg for k in call.keywords} <= {a.arg for a in launch.args.args}


def test_new_session_child_is_preserved_after_reparenting_without_foreign_pid_reuse():
    receipt = {
        "owned_process_group_ids": [10],
        "owner_identities": [{"pid": 10, "startticks": 1}],
    }
    tracked = {}
    table = {
        10: {"pgid": 10, "ppid": 1, "startticks": 1, "state": "S"},
        20: {"pgid": 20, "ppid": 10, "startticks": 2, "state": "S"},
    }
    assert dep.track_descendants(receipt, table, tracked) == [10, 20]
    table = {
        20: {"pgid": 20, "ppid": 1, "startticks": 2, "state": "S"},
        10: {"pgid": 99, "ppid": 1, "startticks": 99, "state": "S"},
        30: {"pgid": 99, "ppid": 10, "startticks": 3, "state": "S"},
    }
    assert dep.track_descendants(receipt, table, tracked) == [20]


def terminal_fixture(tmp_path):
    import hashlib
    import json

    def store(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    cohort = {"members": [{"slot": 4}, {"slot": 5}]}
    cp, cs = store("cohort.json", cohort)
    qp, qs = store(
        "q.json",
        {
            "qualification_ledger_slot": 4,
            "source_commit": dep.SOURCE4,
            "fixed_epochs": 24,
        },
    )
    rp, rs = store("registration.json", {"original": True})
    kp, ks = store(
        "clock.json",
        {
            "schema": "own45-common-original-firstclock-v1",
            "original_training_started_epoch": 1000.0,
        },
    )
    fp, fs = store(
        "failure.json",
        {
            "status": "failed-or-incomplete-preserved-no-retry",
            "started_epoch": 1000.1881328,
            "original_deadline_epoch": 4600.1881328,
        },
    )
    bp, bs = store(
        "baseline-command.json",
        {
            "argv": [
                "python",
                "baseline.py",
                "--seed",
                "20261426",
                "--root",
                str(tmp_path),
                "--started-epoch",
                "1000.1881328",
                "--deadline-epoch",
                "4600.1881328",
                "--qualification-config",
                qp,
                "--qualification-config-sha256",
                qs,
                "--registration",
                rp,
                "--registration-sha256",
                rs,
            ]
        },
    )
    receipt = {
        "schema": "own4-terminal-incomplete-compute-inventory-v1",
        "status": "INCOMPLETE-original-baseline-deadline-expired",
        "qualification_ledger_slot": 4,
        "source_commit": dep.SOURCE4,
        "no_method4_strength_qualification_claim": True,
        "complete_owned_compute_inventory": True,
        "observed_epoch": 4700,
        "owner_identities": [
            {"role": r, "pid": i + 100, "startticks": i + 1}
            for i, r in enumerate(sorted(dep.REQUIRED_ROLES))
        ],
        "owned_process_group_ids": [100],
        "original_cohort": cp,
        "original_cohort_sha256": cs,
        "original_qualification_config": qp,
        "original_qualification_config_sha256": qs,
        "original_registration": rp,
        "original_registration_sha256": rs,
        "original_firstclock_receipt": kp,
        "original_firstclock_receipt_sha256": ks,
        "original_firstclock": 1000.0,
        "failure_artifact_sha256": {fp: fs},
        "expired_baseline_result": fp,
        "expired_baseline_seed": 20261426,
        "expired_baseline_deadline_epoch": 4600.1881328,
        "original_baseline_command": bp,
        "original_baseline_command_sha256": bs,
    }
    return receipt, cohort, sha


def test_real_terminal_inventory_accepts_failed_family_without_invented_ready(tmp_path):
    receipt, cohort, sha = terminal_fixture(tmp_path)
    dep.validate_terminal(receipt, cohort, sha)
    assert "ready" not in receipt and "latency" not in receipt


def test_missing_owner_false_success_and_reset_deadline_are_rejected(tmp_path):
    import copy

    import pytest

    receipt, cohort, sha = terminal_fixture(tmp_path)
    for key, value in [
        ("owner_identities", receipt["owner_identities"][:-1]),
        ("no_method4_strength_qualification_claim", False),
        ("expired_baseline_deadline_epoch", 4700),
        ("complete_owned_compute_inventory", False),
    ]:
        bad = copy.deepcopy(receipt)
        bad[key] = value
        with pytest.raises(AssertionError):
            dep.validate_terminal(bad, cohort, sha)


def test_mutated_failure_artifact_rejected(tmp_path):
    import pytest

    receipt, cohort, sha = terminal_fixture(tmp_path)
    Path(receipt["expired_baseline_result"]).write_text("{}")
    with pytest.raises(AssertionError):
        dep.validate_terminal(receipt, cohort, sha)


def test_distinct_original_baseline_clock_preserved_and_training_clock_substitution_rejected(
    tmp_path,
):
    import pytest

    receipt, cohort, sha = terminal_fixture(tmp_path)
    assert receipt["expired_baseline_deadline_epoch"] != receipt["original_firstclock"] + 3600
    dep.validate_terminal(receipt, cohort, sha)
    receipt["expired_baseline_deadline_epoch"] = receipt["original_firstclock"] + 3600
    with pytest.raises(AssertionError):
        dep.validate_terminal(receipt, cohort, sha)


def test_strict6_and7_compute_owners_block_but_waiting_coordinators_do_not():
    mod = importlib.import_module("post5_scheduling_v2")
    commands = {
        1: "own6_training_controller.py",
        2: "own7_audit_controller.py",
        3: "own6_posttraining.py",
        4: "own6_protect_original45.py",
        5: "own7_parity.py",
        6: "own5_posttraining.py",
    }
    assert mod.extra_compute_busy(commands, 6) == [1, 2, 5]
    assert mod.extra_compute_busy({1: "own6_fresh_cli_replay.py"}, 1) == []
    assert mod.extra_compute_busy({7: "own6_profile_e1_owned.py", 8: "own7_qualify_e1.py"}, 6) == [
        7,
        8,
    ]
