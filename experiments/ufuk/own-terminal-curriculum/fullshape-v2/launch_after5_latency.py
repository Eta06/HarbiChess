"""Launch one bounded source428 profile after genuine original5 latency; no gate changes."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

END = 1791180000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def process_table():
    table = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            pid = int(path.parent.name)
            table[pid] = dict(
                pid=pid, state=fields[0], ppid=int(fields[1]), startticks=int(fields[19])
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return table


def live(identity, table):
    actual = table.get(identity["pid"])
    return bool(
        actual
        and actual["startticks"] == identity["startticks"]
        and actual["state"] not in ("Z", "X")
    )


def track_owned(tracked, table, protected):
    blocked = {item["pid"] for item in protected if live(item, table)}
    while True:
        expanded = blocked | {pid for pid, row in table.items() if row["ppid"] in blocked}
        if expanded == blocked:
            break
        blocked = expanded
    blocked.add(os.getpid())
    roots = {pid for pid, item in tracked.items() if pid not in blocked and live(item, table)}
    while True:
        expanded = roots | {
            pid for pid, row in table.items() if row["ppid"] in roots and pid not in blocked
        }
        if expanded == roots:
            break
        roots = expanded
    for pid in roots:
        tracked[pid] = table[pid]
    for pid in blocked:
        tracked.pop(pid, None)
    return tracked


def send_identity(
    identity,
    sig,
    table_fn=process_table,
    pidfd_open=os.pidfd_open,
    pidfd_send_signal=signal.pidfd_send_signal,
    close=os.close,
):
    if not live(identity, table_fn()):
        return False
    try:
        fd = pidfd_open(identity["pid"])
    except ProcessLookupError:
        return False
    try:
        if not live(identity, table_fn()):
            return False
        pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        close(fd)


def cleanup(tracked, protected):
    track_owned(tracked, process_table(), protected)
    sent = [pid for pid, item in tracked.items() if send_identity(item, signal.SIGTERM)]
    ceiling = time.monotonic() + 5
    while time.monotonic() < ceiling:
        table = process_table()
        track_owned(tracked, table, protected)
        if not any(live(item, table) for item in tracked.values()):
            break
        time.sleep(0.1)
    killed = [pid for pid, item in tracked.items() if send_identity(item, signal.SIGKILL)]
    return dict(
        SIGTERM_pid=sent, SIGKILL_pid=killed, tracked_owned_pid_startticks=list(tracked.values())
    )


def validate_barrier(config):
    immutable = config["immutable_inputs"]
    for path, digest in immutable.items():
        assert sha(path) == digest

    def read_bound(key):
        path = config[key]
        assert path in immutable
        return json.loads(Path(path).read_text())

    cohort = read_bound("original_cohort")
    supplement = read_bound("scheduling_supplement")
    terminal = read_bound("failed4_terminal_receipt")
    activation = read_bound("control_activation_receipt")
    assert cohort["schema"] == "prospective-own45-completion-cohort-v1"
    deadline = cohort["completion_deadline_epoch"]
    assert deadline == config["original_cohort_deadline_epoch"]
    assert supplement["schema"] == "own5-failed4-dependency-analysis-v3-control-supplement-v1"
    assert supplement["qualification_ledger_slot"] == 5
    assert supplement["failed4_terminal_receipt_sha256"] == sha(config["failed4_terminal_receipt"])
    assert terminal["status"] == "INCOMPLETE-original-baseline-deadline-expired"
    assert terminal["qualification_ledger_slot"] == 4
    assert terminal["no_method4_strength_qualification_claim"] is True
    assert terminal["original_cohort_sha256"] == sha(config["original_cohort"])
    assert activation["schema"] == "own5-supplemental-scheduling-control-activation-v1"
    assert activation["control_supplement_sha256"] == sha(config["scheduling_supplement"])
    assert (
        activation["actual_wrapper_sha256"]
        == supplement["supplemental_helper_sha256"]["post5_scheduling_v2.py"]
    )
    assert activation["original_coordinator_sha256"] == config["original5_coordinator_sha256"]
    assert (
        activation["original_qualification_config_sha256"]
        == supplement["original_qualification_config_sha256"]
    )
    assert activation["new_config_sha256"] == supplement["new_config_sha256"]
    assert activation["no_original_deadline_or_gate_change"] is True
    post = Path(config["original5_post_root"])
    assert activation["new_post_root"] == str(post)
    latency = post / "cohort-latency-complete.json"
    completed = post / "latency-process-result.json"
    receipt = json.loads(latency.read_text())
    process = json.loads(completed.read_text())
    assert receipt["schema"] == "own45-latency-owner-completion-receipt-v1"
    assert receipt["slot"] == 5
    assert receipt["source_commit"] == "4515a7c0dda3b4f9615c2fc78a47c872ab14699d"
    assert receipt["coordinator_sha256"] == config["original5_coordinator_sha256"]
    assert receipt["finished_epoch"] <= deadline
    assert receipt["process_receipt_sha256"] == {completed.name: sha(completed)}
    assert process["returncode"] == 0
    assert process["finished_epoch"] <= process["deadline_epoch"] <= deadline
    release_path = post / "failed4-dependency-release.json"
    release = json.loads(release_path.read_text())
    assert release["schema"] == "own4-incomplete-owned-compute-terminated-witness-v1"
    assert release["control_supplement_sha256"] == sha(config["scheduling_supplement"])
    assert release["terminal_receipt_sha256"] == sha(config["failed4_terminal_receipt"])
    assert release["original_cohort_sha256"] == sha(config["original_cohort"])
    assert release["all_registered_owned_groups_and_identities_terminated"] is True
    assert release["method4_ready_latency_final_receipts_fabricated"] is False
    assert release["observed_epoch"] <= deadline
    assert not any(
        live(item, process_table()) for item in release["terminated_tracked_pid_startticks"]
    )
    return {
        "genuine5_latency_receipt_sha256": sha(latency),
        "genuine5_latency_process_sha256": sha(completed),
        "failed4_release_sha256": sha(release_path),
    }


def main():
    if not __debug__:
        raise RuntimeError("Required assertions disabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    assert sha(args.config) == args.config_sha256
    config = json.loads(args.config.read_text())
    root = Path(config["owner_root"])
    root.mkdir(exist_ok=False)
    result = {
        "status": "failed-preserved",
        "schema": "source428-profile-after5-latency-v1",
        "owner_config_sha256": args.config_sha256,
        "waiting_started_epoch": time.time(),
    }
    tracked, owner = {}, None
    protected = config["protected_owner_identities"]
    assert protected and all(item["pid"] > 1 and item["startticks"] > 0 for item in protected)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned profile launcher interruption {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        post = Path(config["original5_post_root"])
        while not (post / "cohort-latency-complete.json").exists():
            if (post / "failure.json").exists():
                raise RuntimeError("Original5 post failed before genuine latency completion")
            if time.time() >= config["latest_profile_start_epoch"]:
                raise TimeoutError("Prospective profile start ceiling exhausted")
            time.sleep(1)
        proof = validate_barrier(config)
        first = time.time()
        assert first < config["latest_profile_start_epoch"]
        deadline = first + 900
        assert deadline < END
        assert "--deadline-epoch" not in config["profile_argv"]
        command = [*config["profile_argv"], "--deadline-epoch", str(deadline)]
        launch = {
            "status": "prospective-source428-profile-launched",
            "argv": command,
            "profile_original_started_epoch": first,
            "profile_original_deadline_epoch": deadline,
            **proof,
        }
        env = {
            **os.environ,
            "PYTHONPATH": config["checkout"] + "/src",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        with (root / "profile.stdout").open("x") as out, (root / "profile.stderr").open("x") as err:
            owner = subprocess.Popen(
                command,
                cwd=config["checkout"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        table = process_table()
        assert owner.pid in table
        tracked[owner.pid] = table[owner.pid]
        launch.update(pid=owner.pid, startticks=table[owner.pid]["startticks"])
        publish(root / "profile-owner.json", launch)
        while owner.poll() is None:
            track_owned(tracked, process_table(), protected)
            if time.time() >= deadline:
                raise TimeoutError("Original900 source428 profile whole ceiling exhausted")
            time.sleep(0.2)
        result.update(
            {key: value for key, value in launch.items() if key != "status"},
            returncode=owner.returncode,
        )
        assert owner.returncode == 0 and time.time() <= deadline
        evidence_path = Path(config["profile_output"]) / "result.json"
        evidence = json.loads(evidence_path.read_text())
        assert (
            evidence["status"]
            == "pass-one-search-acting-v2-development-epoch-and-fullchronological-audit"
        )
        assert evidence["source_commit"] == "428a30f5e1658f3cf159844db547ff0147ade5a9"
        assert evidence["absolute_deadline_epoch"] == deadline
        assert evidence["finished_epoch"] <= deadline
        result.update(
            status="completed-source428-profile-original900", result_sha256=sha(evidence_path)
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["owned_cleanup"] = cleanup(tracked, protected)
        result["finished_epoch"] = time.time()
        publish(root / "result.json", result)


if __name__ == "__main__":
    main()
