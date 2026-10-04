"""Root-owned immutable staging/common-clock launcher; no SSH or outcome selection."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

END = 1791180000
SOURCES = {
    4: "a278bba67bce962cb9294f0d24040e02e9acf1f4",
    5: "4515a7c0dda3b4f9615c2fc78a47c872ab14699d",
}
SEEDS = {4: [20261425, 20261426], 5: [20261525, 20261526]}
PREFIX = {4: "ownv1", 5: "own5"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def no_placeholders(value):
    if isinstance(value, dict):
        for key, item in value.items():
            no_placeholders(key)
            no_placeholders(item)
    elif isinstance(value, list):
        for item in value:
            no_placeholders(item)
    elif isinstance(value, str):
        assert not any(part.startswith("ROOT_") for part in value.split("/"))
        assert value not in (
            "UNKNOWN",
            "UNFROZEN",
            "DRAFT",
            "NOT_FROZEN_DO_NOT_EXECUTE",
        )


def helper(stage, method, suffix):
    return (
        Path(stage)
        / method["helper_subdir"]
        / (PREFIX[method["slot"]] + "_" + suffix + ".py")
    )


def recover_preflight(recovery, now):
    no_placeholders(recovery)
    assert recovery["schema"] == "root-owned-formal45-recovery-v2"
    assert sha(__file__) == recovery["launcher_sha256"]
    for key in (
        "original_launch_config",
        "original_clock",
        "original_paths_config4",
        "original_failure",
    ):
        assert sha(recovery[key]) == recovery[key + "_sha256"]
    config = json.loads(Path(recovery["original_launch_config"]).read_text())
    old = Path(config["stage"])
    assert Path(recovery["original_clock"]) == old / "common-original-firstclock.json"
    clock = json.loads(Path(recovery["original_clock"]).read_text())
    first = validate_clock(clock, recovery["expected_original_firstclock"], now)
    failure = json.loads(Path(recovery["original_failure"]).read_text())
    assert failure["status"] == "failed-preserved-no-retry-no-clock-reset"
    assert failure["firstclock"] == first
    paths4 = json.loads(Path(recovery["original_paths_config4"]).read_text())
    assert Path(recovery["original_paths_config4"]) == old / "paths-config-4.json"
    assert (
        paths4["common_original_firstclock_receipt_sha256"]
        == recovery["original_clock_sha256"]
    )
    assert Path(paths4["common_original_firstclock_receipt"]) == Path(
        recovery["original_clock"]
    )
    baseline_deadlines = {
        r["baseline_deadline_epoch"] for r in paths4["seeds"].values()
    }
    assert len(baseline_deadlines) == 1
    baseline_deadline = baseline_deadlines.pop()
    assert first <= baseline_deadline - 3600 <= now < baseline_deadline < END
    cohort_path = old / "cohort.json"
    assert sha(cohort_path) == paths4["posttraining"]["cohort"]["manifest_sha256"]
    cohort = json.loads(cohort_path.read_text())
    assert cohort["completion_deadline_epoch"] == first + 14520
    assert cohort["slots"] == [4, 5] and cohort["latency_order"] == [4, 5]
    assert not Path(recovery["recovery_root"]).exists()
    for asset in config["assets"]:
        assert sha(old / asset["destination"]) == asset["sha256"]
    assert [m["slot"] for m in config["methods"]] == [4, 5]
    for method in config["methods"]:
        slot = method["slot"]
        assert method["source_commit"] == SOURCES[slot]
        assert method["seeds"] == SEEDS[slot]
        registration_dir = old / ("registration-" + str(slot))
        for name, digest in method["frozen_registration_sha256"].items():
            assert sha(registration_dir / name) == digest
        reg = json.loads((registration_dir / "registration.json").read_text())
        assert (
            reg["source_commit"] == SOURCES[slot]
            and reg["fixed_epochs"] == method["fixed_epochs"]
        )
        assert reg["qualification_ledger_slot"] == slot
        assert reg["earliest_training_epoch"] <= first
        assert (
            reg["whole_training_seconds_per_seed"] == method["whole_training_seconds"]
        )
        assert (
            reg["whole_audit_seconds_from_originalfirstclock"]
            == method["whole_audit_seconds"]
        )
        assert (
            reg["posttraining_reserve_seconds"]
            == method["posttraining_reserve_seconds"]
        )
        assert (
            first
            + method["whole_audit_seconds"]
            + method["posttraining_reserve_seconds"]
            < END
        )
        assert now < first + method["whole_training_seconds"]
        for name, digest in reg["helper_sha256"].items():
            assert sha(old / method["helper_subdir"] / name) == digest
        supplement = recovery["owner_factories"][str(slot)]
        assert Path(supplement["path"]).name.endswith("_owner_config_factory_v2.py")
        assert sha(supplement["path"]) == supplement["sha256"]
        method["paths_config"] = paths4 if slot == 4 else method["paths_config"]
        paths = method["paths_config"]
        paths.update(
            common_original_firstclock_receipt=recovery["original_clock"],
            common_original_firstclock_receipt_sha256=recovery["original_clock_sha256"],
            repo=method["repo"],
            python=config["python"],
        )
        paths["posttraining"]["expected_A100_runtime"] = config["expected_A100_runtime"]
        paths["posttraining"]["cohort"] = paths4["posttraining"]["cohort"]
        member = next(m for m in cohort["members"] if m["slot"] == slot)
        assert member["root"] == paths["posttraining"]["root"]
        assert member["source_commit"] == SOURCES[slot]
        assert member["coordinator_sha256"] == sha(helper(old, method, "posttraining"))
        assert not Path(paths["posttraining"]["root"]).exists()
        for row in paths["seeds"].values():
            if slot == 4:
                assert row["original_training_started_epoch"] == first
                assert row["baseline_deadline_epoch"] == baseline_deadline
            row["original_training_started_epoch"] = first
            row["baseline_deadline_epoch"] = baseline_deadline
            for directory in (
                Path(row["run"]).parent,
                Path(row["audit_controller_result"]).parent,
                Path(row["baseline_root"]),
            ):
                assert not directory.exists(), (
                    f"Prior owner/artifacts exist: {directory}"
                )
    assert sha(config["python"]) == config["python_sha256"]
    return config, first, baseline_deadline


def validate_clock(clock, expected, now):
    assert clock["schema"] == "own45-common-original-firstclock-v1"
    assert clock["slots"] == [4, 5]
    first = clock["original_training_started_epoch"]
    assert type(first) in (int, float) and first == expected and first <= now
    assert first + 22620 < END
    return first


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert sha(args.config) == args.config_sha256
    config = json.loads(args.config.read_text())
    recovery = config
    config, first, baseline_deadline = recover_preflight(recovery, time.time())
    baseline_first = baseline_deadline - 3600
    old_stage = Path(config["stage"])
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "validated-plan-only-no-staging-clock-or-jobs",
                    "source_commits": SOURCES,
                    "seeds": SEEDS,
                }
            )
        )
        return
    # Source/runtime verification is read-only and occurs before the unique cohort clock.
    for method in config["methods"]:
        assert (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=method["repo"], text=True
            ).strip()
            == method["source_commit"]
        )
        assert not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=method["repo"], text=True
        ).strip()
    expected = config["expected_A100_runtime"]
    assert os.uname().nodename == expected["hostname"]
    assert (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        == expected["boot_id"]
    )
    assert expected["python_executable_sha256"] == config["python_sha256"]
    assert (
        subprocess.check_output(
            [config["python"], "-c", "import torch; print(torch.__version__)"],
            text=True,
        ).strip()
        == expected["torch_version"]
    )
    stage = Path(recovery["recovery_root"])
    stage.mkdir(parents=True, exist_ok=False)
    with (
        Path(recovery["original_clock"]).open("rb") as incoming,
        (stage / "common-original-firstclock.json").open("xb") as outgoing,
    ):
        shutil.copyfileobj(incoming, outgoing)
    assert (
        sha(stage / "common-original-firstclock.json")
        == recovery["original_clock_sha256"]
    )
    owned = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Root owned launch interruption {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)

    def launch(name, command, deadline, repo):
        assert time.time() < deadline < END
        publish(
            stage / (name + "-command.json"),
            {
                "argv": list(map(str, command)),
                "original_common_firstclock": first,
                "absolute_deadline_epoch": deadline,
                "launch_epoch": time.time(),
            },
        )
        out = (stage / (name + ".stdout")).open("x")
        err = (stage / (name + ".stderr")).open("x")
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=repo,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(Path(repo) / "src"),
                        *[
                            str(old_stage / method["helper_subdir"])
                            for method in config["methods"]
                        ],
                    ]
                ),
                "PYTHONOPTIMIZE": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        owned.append((name, process, out, err, deadline))
        return process

    def run_config(name, command, repo):
        # Preparation is charged to original first clock, not a new training deadline.
        ceiling = min(first + m["whole_training_seconds"] for m in config["methods"])
        process = launch(name, command, ceiling, repo)
        while process.poll() is None:
            assert time.time() < ceiling
            time.sleep(0.2)
        assert process.returncode == 0

    try:
        cohort_path = old_stage / "cohort.json"
        for method in config["methods"]:
            slot = method["slot"]
            registration = old_stage / ("registration-" + str(slot))
            path = stage / ("paths-config-" + str(slot) + ".json")
            publish(path, method["paths_config"])
            command = [config["python"], recovery["owner_factories"][str(slot)]["path"]]
            for flag, value in (
                ("registration", registration / "registration.json"),
                ("three-input-manifest", registration / "three-input-manifest.json"),
                ("qualification-config", registration / "strength-bindings.json"),
                ("paths-config", path),
            ):
                command += ["--" + flag, value, "--" + flag + "-sha256", sha(value)]
            command += [
                "--helpers",
                old_stage / method["helper_subdir"],
                "--output",
                stage / ("owners-" + str(slot)),
            ]
            run_config("owner-factory-" + str(slot), command, method["repo"])
        # Freeze all generated manifests before ANY baseline game/training child starts.
        publish(
            stage / "frozen-input-inventory.json",
            {
                str(p.relative_to(stage)): sha(p)
                for p in stage.rglob("*.json")
                if p.is_file()
            },
        )
        for method in config["methods"]:
            slot = method["slot"]
            registration = old_stage / ("registration-" + str(slot))
            owners = stage / ("owners-" + str(slot))
            bindings = registration / "strength-bindings.json"
            for seed in SEEDS[slot]:
                row = method["paths_config"]["seeds"][str(seed)]
                train_root = Path(row["run"]).parent
                command = [
                    config["python"],
                    helper(old_stage, method, "training_controller"),
                    "--repo",
                    method["repo"],
                    "--python",
                    config["python"],
                    "--registration",
                    registration / "registration.json",
                    "--registration-sha256",
                    sha(registration / "registration.json"),
                    "--input-manifest",
                    registration / "three-input-manifest.json",
                    "--root",
                    train_root,
                    "--seed",
                    seed,
                    "--started-epoch",
                    first,
                    "--deadline-epoch",
                    first + method["whole_training_seconds"],
                ]
                launch(
                    "train-" + str(seed),
                    command,
                    first + method["whole_training_seconds"],
                    method["repo"],
                )
                manifest = owners / ("audit-manifest-" + str(seed) + ".json")
                command = [
                    config["python"],
                    helper(old_stage, method, "audit_controller"),
                    "--manifest",
                    manifest,
                    "--manifest-sha256",
                    sha(manifest),
                    "--repo",
                    method["repo"],
                    "--python",
                    config["python"],
                    "--audit-script",
                    helper(old_stage, method, "full_audit"),
                    "--audit-script-sha256",
                    sha(helper(old_stage, method, "full_audit")),
                    "--root",
                    Path(row["audit_controller_result"]).parent,
                ]
                launch(
                    "audit-" + str(seed),
                    command,
                    first + method["whole_audit_seconds"],
                    method["repo"],
                )
                post = method["paths_config"]["posttraining"]
                command = [
                    config["python"],
                    helper(old_stage, method, "baseline"),
                    "--qualification-config",
                    bindings,
                    "--qualification-config-sha256",
                    sha(bindings),
                    "--repo",
                    method["repo"],
                    "--python",
                    config["python"],
                    "--registration",
                    registration / "registration.json",
                    "--registration-sha256",
                    sha(registration / "registration.json"),
                    "--weights",
                    post["initial"],
                    "--book",
                    row["book"],
                    "--stockfish",
                    post["stockfish"],
                    "--root",
                    row["baseline_root"],
                    "--seed",
                    seed,
                    "--started-epoch",
                    baseline_first,
                    "--deadline-epoch",
                    baseline_deadline,
                ]
                launch(
                    "baseline-" + str(seed), command, baseline_deadline, method["repo"]
                )
            post_config = owners / "posttraining-config.json"
            launch(
                "post-" + str(slot),
                [
                    config["python"],
                    helper(old_stage, method, "posttraining"),
                    "--config",
                    post_config,
                    "--config-sha256",
                    sha(post_config),
                    "--execute",
                ],
                END - 0.001,
                method["repo"],
            )
        publish(
            stage / "launch-receipt.json",
            {
                "status": "all-four-fixed-training-audit-owners-started",
                "firstclock": first,
                "config_sha256": args.config_sha256,
                "cohort_sha256": sha(cohort_path),
                "scope": "completion only, no scores read",
            },
        )
        remaining = list(owned)
        failed = []
        while remaining:
            for item in list(remaining):
                name, process, out, err, deadline = item
                timed_out = False
                if process.poll() is None:
                    if time.time() < deadline:
                        continue
                    timed_out = True
                    stop(process)
                publish(
                    stage / (name + "-process-result.json"),
                    {
                        "returncode": process.returncode,
                        "finished_epoch": time.time(),
                        "deadline_epoch": deadline,
                        "timed_out": timed_out,
                    },
                )
                if process.returncode != 0 or timed_out:
                    failed.append(name)
                remaining.remove(item)
            time.sleep(0.5)
        publish(
            stage / "completion.json",
            {
                "status": (
                    "all-owned-processes-completed"
                    if not failed
                    else "owned-processes-finished-with-preserved-incomplete-phases"
                ),
                "failed_phases": failed,
                "finished_epoch": time.time(),
                "no_outcome_selection": True,
            },
        )
    except BaseException as error:
        publish(
            stage / "failure.json",
            {
                "status": "failed-preserved-no-retry-no-clock-reset",
                "error": repr(error),
                "finished_epoch": time.time(),
                "firstclock": first,
            },
        )
        raise
    finally:
        for _, process, out, err, _ in owned:
            stop(process)
            out.close()
            err.close()


if __name__ == "__main__":
    main()
