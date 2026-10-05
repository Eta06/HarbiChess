"""Read-only failed-family completion barrier; never substitutes qualification receipts."""

import json
import time
from pathlib import Path

END = 1791180000
SOURCE4 = "a278bba67bce962cb9294f0d24040e02e9acf1f4"
REQUIRED_ROLES = {
    "train-20261425",
    "train-20261426",
    "audit-20261425",
    "audit-20261426",
    "post-4",
    "baseline-20261425",
    "baseline-20261426",
}


def proc_table():
    rows = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            rows[int(path.parent.name)] = {
                "state": fields[0],
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "startticks": int(fields[19]),
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return rows


def identity_live(identity, table):
    row = table.get(identity["pid"])
    return bool(
        row and row["startticks"] == identity["startticks"] and row["state"] not in ("Z", "X")
    )


def remaining_owned(receipt, table):
    groups = set(receipt["owned_process_group_ids"])
    return sorted(
        pid
        for pid, row in table.items()
        if row["state"] not in ("Z", "X")
        and (
            row["pgid"] in groups
            or any(
                item.get("pid") == pid and identity_live(item, table)
                for item in receipt["owner_identities"]
            )
        )
    )


def track_descendants(receipt, table, tracked):
    for item in receipt["owner_identities"]:
        if "pid" in item:
            tracked[(item["pid"], item["startticks"])] = item
    # Capture new sessions while their actual registered ancestors remain alive;
    # keep identities after reparenting. A reused ancestor PID is never followed.
    roots = {pid for (pid, ticks), item in tracked.items() if identity_live(item, table)}
    roots.update(
        pid
        for pid, row in table.items()
        if row["pgid"] in receipt["owned_process_group_ids"] and row["state"] not in ("Z", "X")
    )
    while True:
        children = {
            pid
            for pid, row in table.items()
            if row["ppid"] in roots and row["state"] not in ("Z", "X")
        }
        expanded = roots | children
        if expanded == roots:
            break
        roots = expanded
    for pid in roots:
        item = {"pid": pid, "startticks": table[pid]["startticks"]}
        tracked[(pid, item["startticks"])] = item
    return sorted(pid for (pid, ticks), item in tracked.items() if identity_live(item, table))


def validate_terminal(receipt, cohort, sha):
    assert receipt["schema"] == "own4-terminal-incomplete-compute-inventory-v1"
    assert receipt["status"] == "INCOMPLETE-original-baseline-deadline-expired"
    assert receipt["qualification_ledger_slot"] == 4
    assert receipt["source_commit"] == SOURCE4
    assert receipt["no_method4_strength_qualification_claim"] is True
    assert receipt["complete_owned_compute_inventory"] is True
    assert receipt["observed_epoch"] <= END
    assert {r["role"] for r in receipt["owner_identities"]} >= REQUIRED_ROLES
    assert len({r["role"] for r in receipt["owner_identities"]}) == len(receipt["owner_identities"])
    for item in receipt["owner_identities"]:
        if "pid" in item:
            assert isinstance(item["pid"], int) and item["pid"] > 1
            assert isinstance(item["startticks"], int) and item["startticks"] > 0
        else:
            assert item["role"].startswith("baseline-")
            assert sha(item["terminated_owner_receipt"]) == item["terminated_owner_receipt_sha256"]
            ended = json.loads(Path(item["terminated_owner_receipt"]).read_text())
            assert isinstance(ended["returncode"], int)
            assert ended["finished_epoch"] <= receipt["observed_epoch"]
    assert receipt["owned_process_group_ids"]
    assert all(isinstance(x, int) and x > 1 for x in receipt["owned_process_group_ids"])
    assert receipt["original_cohort_sha256"] == sha(receipt["original_cohort"])
    assert json.loads(Path(receipt["original_cohort"]).read_text()) == cohort
    assert (
        sha(receipt["original_qualification_config"])
        == receipt["original_qualification_config_sha256"]
    )
    assert sha(receipt["original_registration"]) == receipt["original_registration_sha256"]
    q = json.loads(Path(receipt["original_qualification_config"]).read_text())
    assert q["qualification_ledger_slot"] == 4 and q["source_commit"] == SOURCE4
    assert q["fixed_epochs"] == 24
    assert (
        sha(receipt["original_firstclock_receipt"]) == receipt["original_firstclock_receipt_sha256"]
    )
    original_clock = json.loads(Path(receipt["original_firstclock_receipt"]).read_text())
    assert original_clock["schema"] == "own45-common-original-firstclock-v1"
    assert original_clock["original_training_started_epoch"] == receipt["original_firstclock"]
    assert receipt["failure_artifact_sha256"]
    for path, digest in receipt["failure_artifact_sha256"].items():
        assert sha(path) == digest
    failed = json.loads(Path(receipt["expired_baseline_result"]).read_text())
    assert failed["status"] != "completed-frozen-baseline-outcomes-withheld-from-training-decisions"
    assert receipt["expired_baseline_seed"] == 20261426
    assert sha(receipt["original_baseline_command"]) == receipt["original_baseline_command_sha256"]
    command = json.loads(Path(receipt["original_baseline_command"]).read_text())
    argv = command["argv"]

    def argument(flag):
        assert argv.count(flag) == 1
        return argv[argv.index(flag) + 1]

    baseline_first = float(argument("--started-epoch"))
    baseline_deadline = float(argument("--deadline-epoch"))
    assert int(argument("--seed")) == 20261426
    assert Path(argument("--root")) == Path(receipt["expired_baseline_result"]).parent
    assert argument("--qualification-config") == receipt["original_qualification_config"]
    assert (
        argument("--qualification-config-sha256") == receipt["original_qualification_config_sha256"]
    )
    assert argument("--registration") == receipt["original_registration"]
    assert argument("--registration-sha256") == receipt["original_registration_sha256"]
    assert baseline_deadline == baseline_first + 3600
    assert baseline_deadline == receipt["expired_baseline_deadline_epoch"]
    assert failed["started_epoch"] == baseline_first
    assert failed["original_deadline_epoch"] == baseline_deadline
    assert receipt["observed_epoch"] >= receipt["expired_baseline_deadline_epoch"]
    assert (
        sha(receipt["expired_baseline_result"])
        == receipt["failure_artifact_sha256"][receipt["expired_baseline_result"]]
    )


def wait_failed4(
    path,
    digest,
    cohort,
    sha,
    deadline,
    table_fn=proc_table,
    clock=time.time,
    pause=time.sleep,
    tracked_initial=None,
):
    assert sha(path) == digest
    receipt = json.loads(Path(path).read_text())
    validate_terminal(receipt, cohort, sha)
    tracked = dict(tracked_initial or {})
    while True:
        assert sha(path) == digest
        validate_terminal(receipt, cohort, sha)
        table = table_fn()
        if not track_descendants(receipt, table, tracked):
            return {
                "schema": "own4-incomplete-owned-compute-terminated-witness-v1",
                "terminal_receipt_sha256": digest,
                "observed_epoch": clock(),
                "all_registered_owned_groups_and_identities_terminated": True,
                "method4_ready_latency_final_receipts_fabricated": False,
                "terminated_tracked_pid_startticks": list(tracked.values()),
            }
        if clock() >= deadline:
            raise TimeoutError("Failed4 owned compute remains active at original cohort ceiling")
        pause(0.5)
