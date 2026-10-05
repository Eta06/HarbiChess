"""Explicit root execution only; one observed firstclock and seven owned formal8 groups."""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

ADMISSION = 1791169800
END = 1791180000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def proc_table():
    table = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            f = path.read_text().rsplit(")", 1)[1].split()
            pid = int(path.parent.name)
            table[pid] = dict(
                pid=pid, state=f[0], ppid=int(f[1]), pgid=int(f[2]), startticks=int(f[19])
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return table


def descendants(roots, table):
    result = {
        pid: row
        for pid, row in roots.items()
        if table.get(pid, {}).get("startticks") == row["startticks"]
    }
    while True:
        added = {
            pid: row for pid, row in table.items() if pid not in result and row["ppid"] in result
        }
        if not added:
            return result
        result.update(added)


def guard(first, now):
    assert first <= now < END
    assert first <= ADMISSION


def load(path, digest):
    assert sha(path) == digest
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan-template", "owner-template", "planner", "root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-template", "owner-template", "planner"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = load(args.plan_template, args.plan_template_sha256)
    owner = load(args.owner_template, args.owner_template_sha256)
    assert sha(args.planner) == args.planner_sha256
    from late_training_overlap_guard import validate

    late_guard = Path(__file__).with_name("late_training_overlap_guard.py")
    assert sha(late_guard) == plan["late_training_overlap_guard_sha256"]
    validate(plan["late_training_overlap"], sha)
    helpers = Path(plan["factory_inputs"]["helpers"])
    sys.path.insert(0, str(helpers))
    sys.path.insert(0, str(Path(plan["producer_repo"]) / "src"))
    spec = importlib.util.spec_from_file_location("command_planner", args.planner)
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    # Positive actual facts must exist before sampling the formal clock or creating roots.
    import own8_formal_config_factory as factory
    from own8_schedule_v3 import verify_previous

    paths = plan["factory_inputs"]
    factory.check_cli_receipt(
        json.loads(Path(paths["curriculum-qualification"]).read_text()),
        Path(paths["curriculum-qualification-run"]),
        plan["source_commit"],
        sha,
    )
    cases = json.loads(Path(paths["unit-case-inventory"]).read_text())
    assert len(cases) >= 59 and len(set(cases)) == len(cases)
    factory.check_unit_receipt(
        json.loads(Path(paths["unit59-receipt"]).read_text()), plan["source_commit"], cases
    )
    verify_previous(
        {
            "terminal45_barrier_config": paths["terminal45-barrier-config"],
            "terminal45_barrier_config_sha256": sha(paths["terminal45-barrier-config"]),
        },
        sha,
    )
    for name, digest in plan["expected_helper_sha256"].items():
        assert sha(helpers / name) == digest, name
    from own8_prior_failures import validate_descriptor

    validate_descriptor(
        {
            "manifest": paths["standalone8-prior-failures"],
            "manifest_sha256": sha(paths["standalone8-prior-failures"]),
        },
        sha,
    )
    assert not (args.execute and args.validate_only)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": (
                        "static-exact-unit-inventory-CLI600-bindings-valid-full900-still-required"
                    ),
                    "processes_launched": 0,
                    "clocks_sampled": 0,
                    "full900_receipts_present": all(
                        Path(paths[key]).is_file() for key in ("profile-receipt", "auditor-receipt")
                    ),
                }
            )
        )
        return
    profile = json.loads(Path(paths["profile-receipt"]).read_text())
    audit = json.loads(Path(paths["auditor-receipt"]).read_text())
    factory.check_training_evidence(profile, audit)
    factory.actual_model_changed(Path(paths["development-run"]))
    assert len(audit["targeted_actual_data_mutations_rejected"]) == 6
    assert audit["audit_report"]["independently_replayed_mate_certificate_roots"] > 0
    assert audit["audit_report"]["independently_replayed_visited_loss_roots"] > 0
    assert time.time() < ADMISSION
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "positive-proof-preflight-only-no-clocks-no-processes",
                    "source_commit": plan["source_commit"],
                }
            )
        )
        return
    assert not args.root.exists()
    assert not Path(paths["output"]).exists()
    assert not Path(plan["owner_output"]).exists()
    args.root.mkdir(parents=True)
    first = time.time()
    guard(first, first)
    clock = args.root / "common-original-firstclock.json"
    publish(
        clock,
        {
            "schema": "own8-common-original-firstclock-v1",
            "slots": [8],
            "original_training_started_epoch": first,
            "observer_pid": os.getpid(),
            "source_commit": plan["source_commit"],
            "clock_reset": False,
        },
    )
    roots, tracked, pidfds, processes = {}, {}, {}, []
    environment = dict(
        os.environ,
        PYTHONPATH=str(Path(plan["producer_repo"]) / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )

    def scan():
        table = proc_table()
        tracked.update(descendants(roots | tracked, table))
        for pid, row in tracked.items():
            if pid not in pidfds and table.get(pid, {}).get("startticks") == row["startticks"]:
                with suppress(ProcessLookupError):
                    pidfds[pid] = os.pidfd_open(pid)
        return table

    def cleanup():
        table = scan()
        before = list(tracked.values())
        signalled = []
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid, fd in pidfds.items():
                if table.get(pid, {}).get("startticks") != tracked[pid]["startticks"]:
                    continue
                if table[pid]["state"] in ("Z", "X"):
                    continue
                try:
                    signal.pidfd_send_signal(fd, sig)
                    signalled.append(
                        {"pid": pid, "startticks": tracked[pid]["startticks"], "signal": sig.name}
                    )
                except ProcessLookupError:
                    pass
            until = time.time() + 10
            while time.time() < until:
                table = scan()
                live = [
                    pid
                    for pid, row in tracked.items()
                    if table.get(pid, {}).get("startticks") == row["startticks"]
                    and table[pid]["state"] not in ("Z", "X")
                ]
                if not live:
                    break
                time.sleep(0.1)
            if not live:
                break
        return {"tracked_owned": before, "signals": signalled, "remaining_owned": live}

    def launch(name, argv):
        guard(first, time.time())
        publish(
            args.root / f"{name}-command.json",
            {"argv": argv, "original_firstclock": first, "observed_epoch": time.time()},
        )
        with (
            (args.root / f"{name}.stdout").open("x") as stdout,
            (args.root / f"{name}.stderr").open("x") as stderr,
        ):
            process = subprocess.Popen(
                argv,
                cwd=plan["producer_repo"],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        row = proc_table().get(process.pid)
        assert row is not None, f"Owned process identity unavailable: {name}"
        roots[process.pid] = row
        processes.append((name, process))
        scan()
        publish(
            args.root / f"{name}-identity.json", {**row, "name": name, "original_firstclock": first}
        )
        return process

    def setup(name, argv):
        process = launch(name, argv)
        while process.poll() is None:
            scan()
            assert time.time() < ADMISSION, "Fixed03:10 admission ceiling exhausted"
            time.sleep(0.1)
        assert process.returncode == 0, f"{name} failed; no automatic retry"

    def interrupted(signum, _frame):
        raise RuntimeError(f"Owned launch interruption {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        plan["earliest_training_epoch"] = first
        plan["status"] = "actual-positive-proof-bound-root-formal8-plan"
        owner["status"] = "actual-original-firstclock-bound-owner-paths"
        owner["common_original_firstclock_receipt"] = str(clock)
        owner["common_original_firstclock_receipt_sha256"] = sha(clock)
        for row in owner["seeds"].values():
            row["original_training_started_epoch"] = first
            row["baseline_deadline_epoch"] = first + 3600
        planner.complete(owner)
        ownerpath = Path(plan["owner_paths_config"])
        publish(ownerpath, owner)
        publish(args.root / "actual-plan.json", plan)
        setup("factory", planner.factory(plan))
        setup("owner-factory", planner.owners(plan))
        qualification = Path(paths["output"]) / "strength-bindings.json"
        publish(
            args.root / "qualification-config-transfer.json",
            {
                "qualification_config_path": str(qualification),
                "qualification_config_sha256": sha(qualification),
                "qualification_config_base64": base64.b64encode(
                    qualification.read_bytes()
                ).decode(),
                "scope": "exact-public-strength-binding-bytes-only-no-secrets",
            },
        )
        names = ["train25", "audit25", "baseline25", "train26", "audit26", "baseline26", "post"]
        commands = planner.launches(plan)
        assert len(commands) == len(names) == 7
        assert time.time() <= ADMISSION
        for name, argv in zip(names, commands, strict=True):
            launch(name, argv)
        publish(
            args.root / "seven-owners-launched.json",
            {
                "original_firstclock": first,
                "source_commit": plan["source_commit"],
                "fixed_epochs": 8,
                "qualification_config_sha256": sha(qualification),
                "training_deadline": first + 6000,
                "audit_deadline": min(first + 9000, 1791173700),
                "admission_deadline": ADMISSION,
                "hard_end": END,
                "owner_identities": [
                    roots[process.pid] for name, process in processes if name in names
                ],
            },
        )
        while any(process.poll() is None for _, process in processes):
            guard(first, time.time())
            scan()
            assert all(process.poll() in (None, 0) for _, process in processes), (
                "Owned phase failed"
            )
            time.sleep(0.5)
        publish(
            args.root / "result.json",
            {
                "status": "all-owned-processes-finished-no-strength-claim",
                "original_firstclock": first,
                "finished_epoch": time.time(),
                "returncodes": {name: process.returncode for name, process in processes},
            },
        )
    except BaseException as error:
        receipt = cleanup()
        publish(
            args.root / "failed-result.json",
            {
                "status": "failed-INCOMPLETE-preserved-no-retry",
                "error": repr(error),
                "original_firstclock": first,
                "finished_epoch": time.time(),
                "owned_cleanup": receipt,
                "budgets_or_clocks_reset": False,
            },
        )
        raise
    finally:
        for fd in pidfds.values():
            os.close(fd)


if __name__ == "__main__":
    main()
