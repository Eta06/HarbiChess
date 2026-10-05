"""Prospective source428 profile after real families4/5 terminal-owned release; no fake latency."""

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
                pid=pid,
                state=fields[0],
                ppid=int(fields[1]),
                pgid=int(fields[2]),
                startticks=int(fields[19]),
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
        expanded = blocked | {
            pid for pid, row in table.items() if row["ppid"] in blocked
        }
        if expanded == blocked:
            break
        blocked = expanded
    blocked.add(os.getpid())
    roots = {
        pid for pid, item in tracked.items() if pid not in blocked and live(item, table)
    }
    while True:
        expanded = roots | {
            pid
            for pid, row in table.items()
            if row["ppid"] in roots and pid not in blocked
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
    killed = [
        pid for pid, item in tracked.items() if send_identity(item, signal.SIGKILL)
    ]
    return dict(
        SIGTERM_pid=sent,
        SIGKILL_pid=killed,
        tracked_owned_pid_startticks=list(tracked.values()),
    )


def validate_barrier(config):
    from own6_failed4_dependency import remaining_owned
    from own6_failed4_dependency import validate_terminal as validate4
    from own6_failed5_terminal_barrier import bound, validate_release

    immutable = config["immutable_inputs"]
    for path, digest in immutable.items():
        assert sha(path) == digest

    def read(key):
        path = config[key]
        assert path in immutable
        return bound(path, immutable[path], sha)

    cohort = read("original_cohort")
    assert cohort["schema"] == "prospective-own45-completion-cohort-v1"
    assert (
        cohort["completion_deadline_epoch"] == config["original_cohort_deadline_epoch"]
    )
    terminal5 = read("failed5_terminal_receipt")
    release5 = read("failed5_owned_release")
    terminal5_sha = sha(config["failed5_terminal_receipt"])
    validate_release(release5, terminal5, terminal5_sha, sha)
    terminal4 = read("failed4_terminal_receipt")
    validate4(terminal4, cohort, sha)
    release4 = read("failed4_owned_release")
    assert release4["schema"] == "own4-incomplete-owned-compute-terminated-witness-v1"
    assert release4["terminal_receipt_sha256"] == sha(
        config["failed4_terminal_receipt"]
    )
    assert release4["original_cohort_sha256"] == sha(config["original_cohort"])
    assert release4["all_registered_owned_groups_and_identities_terminated"] is True
    assert release4["method4_ready_latency_final_receipts_fabricated"] is False
    assert not remaining_owned(terminal4, process_table())
    assert not any(
        live(row, process_table())
        for row in release4["terminated_tracked_pid_startticks"]
    )
    assert (
        terminal5["original_training_started_epoch"] == terminal4["original_firstclock"]
    )
    return {
        "failed5_terminal_sha256": terminal5_sha,
        "failed5_owned_release_sha256": sha(config["failed5_owned_release"]),
        "failed4_owned_release_sha256": sha(config["failed4_owned_release"]),
        "prior45_qualification_latency_final_receipts_substituted": False,
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
        "schema": "source428-profile-after45-terminal-release-v1",
        "owner_config_sha256": args.config_sha256,
        "waiting_started_epoch": time.time(),
    }
    tracked, owner = {}, None
    protected = config["protected_owner_identities"]
    assert protected and all(
        item["pid"] > 1 and item["startticks"] > 0 for item in protected
    )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned profile launcher interruption {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for key in (
            "failed5_terminal_receipt",
            "failed5_owned_release",
            "failed4_owned_release",
        ):
            while not Path(config[key]).is_file():
                if time.time() >= config["latest_profile_start_epoch"]:
                    raise TimeoutError("Prospective profile start ceiling exhausted")
                time.sleep(0.5)
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
        with (
            (root / "profile.stdout").open("x") as out,
            (root / "profile.stderr").open("x") as err,
        ):
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
                raise TimeoutError(
                    "Original900 source428 profile whole ceiling exhausted"
                )
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
            status="completed-source428-profile-original900",
            result_sha256=sha(evidence_path),
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
