"""Owner-invoked protection monitor; only explicitly registered method6 PID identities."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

from own6_audit_support import publish, sha
from own6_previous_methods_barrier import repaired5_member, verify_failed4_release

END = 1791180000


def process_table(proc=Path("/proc")):
    table = {}
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            table[int(directory.name)] = {
                "pid": int(directory.name),
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "startticks": int(fields[19]),
                "state": fields[0],
            }
        except (
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
            IndexError,
            ValueError,
        ):
            continue
    return table


def own_descendants(owners, table, protected=()):
    blocked = {pid for pid, row in table.items() if (pid, row["startticks"]) in protected}
    while True:
        expanded = blocked | {pid for pid, row in table.items() if row["ppid"] in blocked}
        if expanded == blocked:
            break
        blocked = expanded
    table = {pid: row for pid, row in table.items() if pid not in blocked}
    owned = {
        row["pid"]: row
        for row in owners
        if row["pid"] in table and table[row["pid"]]["startticks"] == row["startticks"]
    }
    changed = True
    while changed:
        changed = False
        for pid, row in table.items():
            if pid not in owned and row["ppid"] in owned:
                owned[pid] = row
                changed = True
    return owned


def stop_targets(tracked, current, send, sig):
    signalled = []
    for pid, identity in tracked.items():
        actual = current.get(pid)
        if actual is not None and actual["startticks"] == identity["startticks"]:
            try:
                send(pid, sig)
            except ProcessLookupError:
                continue
            signalled.append(pid)
    return signalled


def identity_sender(tracked):
    def send(pid, sig):
        # pidfd fixes identity across the /proc-check-to-signal interval.
        fd = os.pidfd_open(pid)
        try:
            current = process_table().get(pid)
            if current is None or current["startticks"] != tracked[pid]["startticks"]:
                return
            signal.pidfd_send_signal(fd, sig)
        finally:
            os.close(fd)

    return send


def old_both_ready(config):
    path = Path(config["original_cohort"])
    assert sha(path) == config["original_cohort_sha256"]
    cohort = json.loads(path.read_text())
    assert cohort["slots"] == [4, 5]
    proof = {}
    members = cohort["members"]
    if "failed4_dependency_supplement" in config:
        supplement, terminal, only5 = repaired5_member(config, cohort, sha)
        release_path = Path(only5["root"]) / "failed4-dependency-release.json"
        try:
            release = json.loads(release_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        verify_failed4_release(
            config["failed4_dependency_supplement"], release, supplement, terminal, sha
        )
        assert release["observed_epoch"] <= cohort["completion_deadline_epoch"]
        proof[str(release_path)] = sha(release_path)
        members = [only5]
    for member in members:
        root = Path(member["root"])
        p = root / "cohort-ready.json"
        try:
            ready = json.loads(p.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        assert ready["slot"] == member["slot"]
        assert ready["source_commit"] == member["source_commit"]
        assert ready["coordinator_sha256"] == member["coordinator_sha256"]
        assert ready["finished_epoch"] <= cohort["completion_deadline_epoch"]
        assert set(ready["process_receipt_sha256"]) == set(member["required_process_receipt_names"])
        for filename, digest in ready["process_receipt_sha256"].items():
            assert Path(filename).name == filename and sha(root / filename) == digest
            result = json.loads((root / filename).read_text())
            assert (
                result["returncode"] == 0 and result["finished_epoch"] <= result["deadline_epoch"]
            )
        proof[str(p)] = sha(p)
    return proof


def prelatency_complete(config):
    proof = {}
    for info in config["required_own6_completion_receipts"]:
        path = Path(info["path"])
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        for key, expected in info["required_fields"].items():
            assert data[key] == expected
        if "finished_epoch" in data and "deadline_epoch" in data:
            assert data["finished_epoch"] <= data["deadline_epoch"]
        proof[str(path)] = sha(path)
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert sha(args.config) == args.config_sha256
    config = json.loads(args.config.read_text())
    assert config["schema"] == "own6-protect-original45-owned-compute-v1"
    assert config["poll_seconds"] == 0.5
    owners = config["owned6_owner_identities"]
    assert hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
    assert owners and all(row["pid"] > 1 and row["startticks"] > 0 for row in owners)
    protected = {(r["pid"], r["startticks"]) for r in config["protected45_owner_identities"]}
    assert not {(r["pid"], r["startticks"]) for r in owners} & protected
    assert len(config["required_own6_completion_receipts"]) == 9
    assert (
        len({str(Path(r["path"]).resolve()) for r in config["required_own6_completion_receipts"]})
        == 9
    )
    # Two trainings, two full audits, two fresh replays, CUDA parity, two baselines.
    assert config["phase_counts"] == {
        "training": 2,
        "audit": 2,
        "replay": 2,
        "cuda": 1,
        "baseline": 2,
    }
    if not args.execute:
        print(json.dumps({"status": "validated-protection-plan-only-no-signals"}))
        return
    args.root.mkdir(parents=True, exist_ok=False)
    tracked = {}
    assert os.getpid() not in {r["pid"] for r in owners}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned monitor interruption {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while time.time() < END:
            table = process_table()
            tracked.update(own_descendants(owners, table, protected))
            assert not {(r["pid"], r["startticks"]) for r in tracked.values()} & protected
            complete = prelatency_complete(config)
            if complete is not None:
                publish(
                    args.root / "protection-completed.json",
                    {
                        "status": "all-fixedE8-prelatency-compute-and-baselines-complete",
                        "completion_receipt_sha256": complete,
                        "finished_epoch": time.time(),
                    },
                )
                return
            ready = old_both_ready(config)
            if ready is not None:
                tracked.update(own_descendants(owners, process_table(), protected))
                signalled = stop_targets(
                    tracked, process_table(), identity_sender(tracked), signal.SIGTERM
                )
                cleanup_deadline = time.monotonic() + 5
                while time.monotonic() < cleanup_deadline:
                    live = process_table()
                    if not any(
                        pid in live
                        and live[pid]["startticks"] == r["startticks"]
                        and live[pid]["state"] != "Z"
                        for pid, r in tracked.items()
                    ):
                        break
                    time.sleep(0.1)
                killed = stop_targets(
                    tracked, process_table(), identity_sender(tracked), signal.SIGKILL
                )
                publish(
                    args.root / "method6-INCOMPLETE-protection-stop.json",
                    {
                        "status": (
                            "method6-INCOMPLETE-protection-stop-no-earlier-checkpoint-selection"
                        ),
                        "reason": (
                            "original45-both-ready-before-owned6-"
                            "prelatency-compute-baselines-complete"
                        ),
                        "original45_ready_receipt_sha256": ready,
                        "owned_pid_startticks": list(tracked.values()),
                        "SIGTERM_pid": signalled,
                        "SIGKILL_pid": killed,
                        "finished_epoch": time.time(),
                        "preserved_last_native_directories": config["native_checkpoint_roots"],
                        "no_old_process_signalled": True,
                        "no_budget_reset": True,
                    },
                )
                return
            time.sleep(0.5)
        raise TimeoutError("Original06UTC physical lease exhausted")
    except BaseException as error:
        # Losing the safety monitor cannot leave method6 compute unprotected.
        stop_targets(tracked, process_table(), identity_sender(tracked), signal.SIGTERM)
        time.sleep(0.2)
        stop_targets(tracked, process_table(), identity_sender(tracked), signal.SIGKILL)
        publish(
            args.root / "protection-monitor-failure.json",
            {
                "error": repr(error),
                "status": "INCOMPLETE-monitor-failure-preserved",
                "finished_epoch": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
