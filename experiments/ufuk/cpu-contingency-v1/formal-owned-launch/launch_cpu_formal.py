"""Separate CPU fixed-E8 launcher; no A100 receipt reuse or deadline reset."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

SOURCE = "3be5b87db27a0fbde83464e7ea7157f0d9a76ae4"
SEEDS = (20261925, 20261926)
END = 1791180000
ADMISSION = 1791173100
AUDIT_HELPERS = {
    f"cpu_contingency_{name}.py"
    for name in (
        "adapter_controls",
        "audit_support",
        "audit_core",
        "full_audit",
        "fresh_cli_replay",
        "qualify_e1",
    )
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def ticks(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2 :].split()
        return int(fields[19]), fields[0]
    except FileNotFoundError:
        return None


def clock(first, now):
    assert first <= now < ADMISSION and now < END
    return min(first + 4500, 1791177900), min(first + 4800, 1791178200)


def preflight(cfg):
    assert cfg["schema"] == "cpu-fixedE8-owned-launch-config-v1"
    assert cfg["source_commit"] == SOURCE
    checkout = Path(cfg["checkout"])
    helpers = Path(cfg["helpers"])
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        == SOURCE
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=checkout, text=True
    ).strip()
    assert read(cfg["formal_protocol"]["path"])["source_commit"] == SOURCE
    for key in ("formal_protocol", "CLI_result", "profile_result", "E1_audit_result"):
        assert sha(cfg[key]["path"]) == cfg[key]["sha256"]
    cli = read(cfg["CLI_result"]["path"])
    assert cli["source_commit"] == SOURCE and cli["status"] == "pass"
    assert len(cli["byte_exact_artifacts"]) == 20
    assert cli["both_all_native_epochs_freshprocess_strictload"] == [0, 1, 2]
    assert cli["both_final_full_native_freshprocess_strictload"] is True
    profile = read(cfg["profile_result"]["path"])
    proof = read(cfg["E1_audit_result"]["path"])
    assert profile["status"] == "pass-actual-CPU-E1-qualified"
    assert profile["source_commit"] == proof["source_commit"] == SOURCE
    assert profile["audit_sha256"] == cfg["E1_audit_result"]["sha256"]
    assert profile["finished_epoch"] <= profile["absolute_deadline_epoch"]
    assert (
        proof["finished_epoch"]
        <= proof["absolute_deadline_epoch"]
        == profile["absolute_deadline_epoch"]
    )
    assert proof["status"] == (
        "pass-actualCPU-search-acting-v4-E1-all-data-FIRST8-LAST8-"
        "original64-masks-raw-packets-and-mutations"
    )
    assert proof["torch_version"] == "2.14.1+cpu"
    assert proof["qualified_production_core_sha256"] == sha(
        helpers / "cpu_contingency_audit_core.py"
    )
    assert proof["qualified_report_guard_sha256"] == sha(
        helpers / "cpu_contingency_adapter_controls.py"
    )
    assert len(proof["targeted_actual_data_mutations_rejected"]) == 6 and all(
        proof["targeted_actual_data_mutations_rejected"].values()
    )
    report = proof["audit_report"]
    assert (
        report["raw_actor_replayed"] == 16384
        and report["actor_count"] == 64
        and report["epoch"] == 1
    )
    assert (
        report["neural_witness_K"] == 8
        and report["prescribed_neural_roots_verified"]
        == report["raw_actor_packet_roots_verified"]
        == 18
    )
    assert (
        report["optimizer_committed"] > 0
        and report["independently_replayed_visited_loss_roots"] > 0
        and report["independently_replayed_mate_certificate_roots"] > 0
    )
    assert set(cfg["helper_sha256"]) >= AUDIT_HELPERS
    for name, digest in cfg["helper_sha256"].items():
        assert Path(name).name == name and sha(helpers / name) == digest
    assert [row["seed"] for row in cfg["seeds"]] == list(SEEDS)
    for row in cfg["seeds"]:
        inputs = row["inputs"]
        assert set(inputs) == {"initial_weights", "book", "experiment_config", "protocol"}
        for item in inputs.values():
            assert sha(item["path"]) == item["sha256"]
        assert inputs["protocol"] == cfg["formal_protocol"]
        frozen = read(inputs["experiment_config"]["path"])
        assert (
            frozen["seed"] == row["seed"]
            and frozen["device"] == "cpu"
            and frozen["actors"]["games"] == 64
            and frozen["epoch_steps"] == 256
        )
        assert (
            not Path(row["run"]).exists()
            and not Path(row["audit_root"]).exists()
            and not Path(row["replay_run"]).exists()
        )
    return profile


def producer_command(cfg, row, deadline, resume=False):
    cmd = [cfg["python"], "-m", "harbichess.training.torch_search_acting_run"]
    for arg, key in [
        ("weights", "initial_weights"),
        ("book", "book"),
        ("config", "experiment_config"),
        ("protocol", "protocol"),
    ]:
        cmd += ["--" + arg, row["inputs"][key]["path"]]
    cmd += [
        row["run"],
        "--source-commit",
        SOURCE,
        "--max-epochs",
        "8",
        "--checkpoint-interval",
        "1",
        "--deadline-epoch",
        str(deadline),
        "--memory-max-bytes",
        str(16 * 1024**3),
        "--disk-min-free-bytes",
        str(256 * 1024**2),
    ]
    return cmd + (
        ["--resume", str(Path(row["run"]) / "checkpoints/epoch-00000001")]
        if resume
        else ["--stop-at", "1"]
    )


def track_group(child, identity):
    """Capture group descendants only while the original leader identity is present."""
    live = ticks(child.pid)
    if live is None or live[0] != identity:
        return {}
    found = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = path.read_text()
            fields = text[text.rfind(")") + 2 :].split()
            if int(fields[2]) == child.pid and int(fields[19]) >= identity:
                found[int(path.parent.name)] = int(fields[19])
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def stop_owned(child, identity, tracked):
    tracked.update(track_group(child, identity))
    for pid, expected in tracked.items():
        live = ticks(pid)
        if live is None:
            continue
        if live[0] != expected:
            raise RuntimeError("owned PID reused; refusing signal")
        with suppress(ProcessLookupError):
            fd = os.pidfd_open(pid)
            try:
                if ticks(pid) is not None and ticks(pid)[0] == expected:
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
            finally:
                os.close(fd)
    if child.poll() is None:
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            live = ticks(child.pid)
            assert live is not None and live[0] == identity
            fd = os.pidfd_open(child.pid)
            try:
                assert ticks(child.pid)[0] == identity
                signal.pidfd_send_signal(fd, signal.SIGKILL)
                child.wait(timeout=3)
            finally:
                os.close(fd)
    for pid, expected in tracked.items():
        live = ticks(pid)
        if live is not None and live[0] == expected and live[1] != "Z":
            with suppress(ProcessLookupError):
                fd = os.pidfd_open(pid)
                try:
                    if ticks(pid) is not None and ticks(pid)[0] == expected:
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                finally:
                    os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert sha(args.config) == args.config_sha256
    cfg = read(args.config)
    preflight(cfg)
    if not args.execute:
        print(
            json.dumps(
                {"status": "validated-no-clock-no-process", "config_sha256": sha(args.config)}
            )
        )
        return
    first = time.time()
    train_end, audit_end = clock(first, first)
    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=False)
    publish(
        root / "original-common-clock.json",
        dict(
            original_first_epoch=first,
            train_deadline=train_end,
            audit_deadline=audit_end,
            hard_end_epoch=END,
        ),
    )
    helpers = Path(cfg["helpers"])
    sys.path.insert(0, str(helpers))
    from cpu_contingency_audit_support import guard

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(Path(cfg["checkout"]) / "src") + ":" + str(helpers),
    )
    owners = []
    tracked = {}
    states = {}
    result = dict(
        schema="cpu-fixedE8-owned-cohort-v1",
        status="incomplete",
        source_commit=SOURCE,
        original_first_epoch=first,
        train_deadline=train_end,
        audit_deadline=audit_end,
        config_sha256=sha(args.config),
    )

    admitting = True

    def launch(name, cmd, deadline):
        if admitting:
            assert time.time() < ADMISSION, "initial owner admission ceiling expired"
        guard(deadline, root)
        with (
            (root / (name + ".stdout.log")).open("x") as out,
            (root / (name + ".stderr.log")).open("x") as err,
        ):
            if admitting:
                assert time.time() < ADMISSION, "initial pre-Popen admission expired"
            child = subprocess.Popen(
                cmd,
                cwd=cfg["checkout"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        identity = ticks(child.pid)
        assert identity is not None
        owners.append((name, child, identity[0], deadline))
        tracked[child.pid] = {child.pid: identity[0]}
        publish(
            root / (name + ".owner.json"),
            dict(
                command=cmd,
                pid=child.pid,
                startticks=identity[0],
                pgid=child.pid,
                started_epoch=time.time(),
                original_common_first_epoch=first,
                deadline_epoch=deadline,
            ),
        )
        return child

    try:
        for row in cfg["seeds"]:
            seed = row["seed"]
            clock(first, time.time())
            audit_root = Path(row["audit_root"])
            audit_root.mkdir(parents=True, exist_ok=False)
            spec = dict(
                schema="ufuk-cpu-contingency-audit-manifest-v1",
                status="frozen-before-formal-execution",
                protocol_id="cpu-contingency-v1",
                qualification_ledger_slot=8,
                source_commit=SOURCE,
                producer_checkout=cfg["checkout"],
                run=row["run"],
                fixed_epochs=8,
                neural_audit_epochs=[1, 2, 3, 5, 7, 8],
                neural_witness_K=8,
                frozen_config=read(row["inputs"]["experiment_config"]["path"]),
                inputs=row["inputs"],
                helper_sha256={name: cfg["helper_sha256"][name] for name in AUDIT_HELPERS},
                original_training_started_epoch=first,
                original_training_deadline_epoch=train_end,
                whole_training_seconds=train_end - first,
                whole_audit_seconds=4800,
                absolute_audit_cutoff_epoch=1791178200,
            )
            manifest = audit_root / "audit-manifest.json"
            publish(manifest, spec)
            child = launch(f"train-{seed}-pause1", producer_command(cfg, row, train_end), train_end)
            audit = launch(
                f"audit-{seed}",
                [
                    cfg["python"],
                    str(helpers / "cpu_contingency_full_audit.py"),
                    "--manifest",
                    str(manifest),
                    "--manifest-sha256",
                    sha(manifest),
                    "--run",
                    row["run"],
                    "--output",
                    str(audit_root / "full-audit-result.json"),
                    "--deadline-epoch",
                    str(audit_end),
                ],
                audit_end,
            )
            states[seed] = dict(
                row=row,
                producer=child,
                auditor=audit,
                resumed=False,
                replay=None,
                manifest=manifest,
            )
        admitting = False
        publish(
            root / "launch-progress.json",
            dict(status="both-producers-and-auditors-started", original_first_epoch=first),
        )
        while True:
            guard(min(audit_end, END), root)
            for name, child, owner_identity, deadline in owners:
                tracked[child.pid].update(track_group(child, owner_identity))
                if child.poll() is None and time.time() >= deadline:
                    raise TimeoutError(name + " original deadline exhausted")
                if child.poll() is not None and child.returncode != 0:
                    raise RuntimeError(name + " returned " + str(child.returncode))
            replay_busy = any(
                s["replay"] is not None and s["replay"].poll() is None for s in states.values()
            )
            for seed, state in states.items():
                row = state["row"]
                run = Path(row["run"])
                if not state["resumed"] and state["producer"].poll() == 0:
                    assert (run / "checkpoints/epoch-00000001/checkpoint.json").is_file()
                    state["producer"] = launch(
                        f"train-{seed}-fresh-resume8",
                        producer_command(cfg, row, train_end, True),
                        train_end,
                    )
                    state["resumed"] = True
                if (
                    not replay_busy
                    and state["replay"] is None
                    and (run / "checkpoints/epoch-00000002/checkpoint.json").is_file()
                ):
                    replay_end = min(time.time() + 600, END - 1)
                    state["replay"] = launch(
                        f"replay-{seed}",
                        [
                            cfg["python"],
                            str(helpers / "cpu_contingency_fresh_cli_replay.py"),
                            "--manifest",
                            str(state["manifest"]),
                            "--manifest-sha256",
                            sha(state["manifest"]),
                            "--run",
                            row["run"],
                            "--audit-run",
                            row["replay_run"],
                            "--deadline-epoch",
                            str(replay_end),
                        ],
                        replay_end,
                    )
                    replay_busy = True
            if all(
                s["resumed"]
                and s["producer"].poll() == 0
                and s["auditor"].poll() == 0
                and s["replay"] is not None
                and s["replay"].poll() == 0
                for s in states.values()
            ):
                for _seed, s in states.items():
                    assert (
                        read(Path(s["row"]["run"]) / "checkpoints/epoch-00000008/actor.json")[
                            "epoch"
                        ]
                        == 8
                    )
                    audit = read(Path(s["row"]["audit_root"]) / "full-audit-result.json")
                    assert audit["audited_native_checkpoints"] == 9
                result["status"] = "completed-both-fixedCPU8-fullaudits-and-independent-replays"
                break
            time.sleep(0.25)
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        cleanup_errors = []
        for _name, child, identity, _deadline in owners:
            try:
                stop_owned(child, identity, tracked[child.pid])
            except BaseException as exc:
                cleanup_errors.append(dict(name=_name, error=repr(exc)))
        if cleanup_errors:
            result.update(status="incomplete-owned-cleanup-failure", cleanup_errors=cleanup_errors)
        result.update(
            finished_epoch=time.time(),
            owners=[
                dict(
                    name=name,
                    pid=child.pid,
                    startticks=identity,
                    returncode=child.returncode,
                    deadline_epoch=deadline,
                )
                for name, child, identity, deadline in owners
            ],
        )
        publish(root / "result.json", result)


if __name__ == "__main__":
    main()
