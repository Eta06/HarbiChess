"""Synthetic barrier/ownership regressions; no process, SSH or CUDA execution."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "launcher", Path(__file__).with_name("launch_after5_latency.py")
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)

    def store(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return str(path)

    cohort = store(
        "cohort.json",
        {"schema": "prospective-own45-completion-cohort-v1", "completion_deadline_epoch": 100},
    )
    terminal = store(
        "terminal.json",
        {
            "status": "INCOMPLETE-original-baseline-deadline-expired",
            "qualification_ledger_slot": 4,
            "no_method4_strength_qualification_claim": True,
            "original_cohort_sha256": mod.sha(cohort),
        },
    )
    supplement = store(
        "supplement.json",
        {
            "schema": "own5-failed4-dependency-analysis-v3-control-supplement-v1",
            "qualification_ledger_slot": 5,
            "failed4_terminal_receipt_sha256": mod.sha(terminal),
            "supplemental_helper_sha256": {"post5_scheduling_v2.py": "a" * 64},
            "original_qualification_config_sha256": "b" * 64,
            "new_config_sha256": "c" * 64,
        },
    )
    post = tmp_path / "post"
    post.mkdir()
    activation = store(
        "activation.json",
        {
            "schema": "own5-supplemental-scheduling-control-activation-v1",
            "control_supplement_sha256": mod.sha(supplement),
            "actual_wrapper_sha256": "a" * 64,
            "original_coordinator_sha256": "d" * 64,
            "original_qualification_config_sha256": "b" * 64,
            "new_config_sha256": "c" * 64,
            "no_original_deadline_or_gate_change": True,
            "new_post_root": str(post),
        },
    )
    process = post / "latency-process-result.json"
    process.write_text(json.dumps({"returncode": 0, "finished_epoch": 90, "deadline_epoch": 95}))
    (post / "cohort-latency-complete.json").write_text(
        json.dumps(
            {
                "schema": "own45-latency-owner-completion-receipt-v1",
                "slot": 5,
                "source_commit": "4515a7c0dda3b4f9615c2fc78a47c872ab14699d",
                "coordinator_sha256": "d" * 64,
                "finished_epoch": 91,
                "process_receipt_sha256": {process.name: mod.sha(process)},
            }
        )
    )
    (post / "failed4-dependency-release.json").write_text(
        json.dumps(
            {
                "schema": "own4-incomplete-owned-compute-terminated-witness-v1",
                "control_supplement_sha256": mod.sha(supplement),
                "terminal_receipt_sha256": mod.sha(terminal),
                "original_cohort_sha256": mod.sha(cohort),
                "all_registered_owned_groups_and_identities_terminated": True,
                "method4_ready_latency_final_receipts_fabricated": False,
                "observed_epoch": 80,
                "terminated_tracked_pid_startticks": [],
            }
        )
    )
    config = {
        "original_cohort": cohort,
        "scheduling_supplement": supplement,
        "failed4_terminal_receipt": terminal,
        "control_activation_receipt": activation,
        "original_cohort_deadline_epoch": 100,
        "original5_post_root": str(post),
        "original5_coordinator_sha256": "d" * 64,
        "immutable_inputs": {p: mod.sha(p) for p in (cohort, terminal, supplement, activation)},
    }
    return config, post


def test_bound_real5_latency_and_failed4_metadata_accepts_without_fake4_receipts(tmp_path):
    config, post = fixture(tmp_path)
    assert mod.validate_barrier(config)["failed4_release_sha256"]
    assert not (tmp_path / "method4-latency").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("finished_epoch", 101),
        ("slot", 4),
        ("source_commit", "wrong"),
        ("coordinator_sha256", "e" * 64),
    ],
)
def test_bad_latency_binding_rejected(tmp_path, field, value):
    config, post = fixture(tmp_path)
    path = post / "cohort-latency-complete.json"
    value_map = json.loads(path.read_text())
    value_map[field] = value
    path.write_text(json.dumps(value_map))
    with pytest.raises(AssertionError):
        mod.validate_barrier(config)


def test_extra_latency_process_and_forged_failure_release_rejected(tmp_path):
    config, post = fixture(tmp_path)
    path = post / "cohort-latency-complete.json"
    value = json.loads(path.read_text())
    value["process_receipt_sha256"]["extra.json"] = "a" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        mod.validate_barrier(config)
    config, post = fixture(tmp_path / "second")
    path = post / "failed4-dependency-release.json"
    value = json.loads(path.read_text())
    value["method4_ready_latency_final_receipts_fabricated"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        mod.validate_barrier(config)


def test_reused_pid_before_and_during_pidfd_open_never_signalled():
    identity = {"pid": 10, "startticks": 1}
    reused = {10: {"pid": 10, "startticks": 2, "state": "S"}}
    sent, opened, closed = [], [], []
    assert not mod.send_identity(
        identity,
        15,
        lambda: reused,
        lambda pid: opened.append(pid),
        lambda fd, sig: sent.append(fd),
        lambda fd: closed.append(fd),
    )
    assert not opened and not sent
    old = {10: {"pid": 10, "startticks": 1, "state": "S"}}
    states = iter((old, reused))
    assert not mod.send_identity(
        identity,
        15,
        lambda: next(states),
        lambda pid: 88,
        lambda fd, sig: sent.append(fd),
        lambda fd: closed.append(fd),
    )
    assert not sent and closed == [88]


def test_new_session_descendant_tracked_and_protected_branch_pruned():
    table = {
        10: {"pid": 10, "startticks": 1, "state": "S", "ppid": 1},
        20: {"pid": 20, "startticks": 2, "state": "S", "ppid": 10},
        30: {"pid": 30, "startticks": 3, "state": "S", "ppid": 20},
        40: {"pid": 40, "startticks": 4, "state": "S", "ppid": 10},
    }
    tracked = {10: table[10], 20: table[20]}
    mod.track_owned(tracked, table, [table[20]])
    assert set(tracked) == {10, 40}
    # A captured child persists after its parent exits and child is reparented.
    orphan = {40: {**table[40], "ppid": 1}}
    mod.track_owned(tracked, orphan, [])
    assert mod.live(tracked[40], orphan)


def test_publish_once_complete_json_and_no_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    mod.publish(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": True}
    with pytest.raises(FileExistsError):
        mod.publish(path, {"complete": False})
    assert json.loads(path.read_text()) == {"complete": True}
