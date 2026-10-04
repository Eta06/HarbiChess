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


def factory_command(python, stage, method, first):
    command = [python, str(helper(stage, method, "formal_config_factory"))]
    for key, value in method["factory_args"].items():
        command += ["--" + key, str(value)]
    command += [
        "--helpers",
        str(Path(stage) / method["helper_subdir"]),
        "--output",
        str(Path(stage) / ("registration-" + str(method["slot"]))),
        "--fixed-epochs",
        str(method["fixed_epochs"]),
        "--whole-training-seconds",
        str(method["whole_training_seconds"]),
        "--whole-audit-seconds",
        str(method["whole_audit_seconds"]),
        "--posttraining-reserve-seconds",
        str(method["posttraining_reserve_seconds"]),
        "--earliest-training-epoch",
        str(first),
        "--freeze",
    ]
    return command


def preflight(config, now):
    no_placeholders(config)
    assert config["schema"] == "root-owned-formal45-launch-v1"
    assert config["status"] == "frozen-before-any-formal45-execution"
    assert sha(__file__) == config["launcher_sha256"]
    assert [m["slot"] for m in config["methods"]] == [4, 5]
    stage = Path(config["stage"])
    assert stage.is_absolute() and not stage.exists()
    assert stage.is_relative_to("/content/harbichess-fullgame-method2-inputs")
    assert sha(config["python"]) == config["python_sha256"]
    destinations = set()
    for item in config["assets"]:
        relative = Path(item["destination"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert relative.as_posix() not in destinations
        destinations.add(relative.as_posix())
        assert not Path(item["source"]).is_symlink()
        assert sha(item["source"]) == item["sha256"]
    totals = []
    for method in config["methods"]:
        slot = method["slot"]
        assert method["source_commit"] == SOURCES[slot]
        assert method["seeds"] == SEEDS[slot]
        assert method["fixed_epochs"] == (24 if slot == 4 else 8)
        assert 0 < method["whole_training_seconds"] <= method["whole_audit_seconds"]
        total = method["whole_audit_seconds"] + method["posttraining_reserve_seconds"]
        totals.append(total)
        assert now + total < END
        # Exact original helper versions must implement persisted earliest-clock guards.
        names = {item["destination"]: item for item in config["assets"]}
        for suffix in (
            "formal_config_factory",
            "owner_config_factory",
            "training_controller",
        ):
            relative = str(
                Path(method["helper_subdir"]) / (PREFIX[slot] + "_" + suffix + ".py")
            )
            text = Path(names[relative]["source"]).read_text()
            assert "earliest_training_epoch" in text
        if "frozen_registration_directory" in method:
            frozen = Path(method["frozen_registration_directory"])
            inventory = method["frozen_registration_sha256"]
            assert len(inventory) == 6
            for name, digest in inventory.items():
                assert Path(name).name == name
                assert sha(frozen / name) == digest
            registration = json.loads((frozen / "registration.json").read_text())
            assert registration["status"] == "frozen-before-formal-execution"
            assert registration["source_commit"] == SOURCES[slot]
            assert registration["qualification_ledger_slot"] == slot
            assert registration["fixed_epochs"] == method["fixed_epochs"]
            assert registration["earliest_training_epoch"] <= now
            assert (
                registration["whole_training_seconds_per_seed"]
                == method["whole_training_seconds"]
            )
            assert (
                registration["whole_audit_seconds_from_originalfirstclock"]
                == method["whole_audit_seconds"]
            )
            assert (
                registration["posttraining_reserve_seconds"]
                == method["posttraining_reserve_seconds"]
            )
            for filename, digest in registration["helper_sha256"].items():
                relative = str(Path(method["helper_subdir"]) / filename)
                assert names[relative]["sha256"] == digest
            assert (
                sha(frozen / "three-input-manifest.json")
                == registration["three_input_manifest_sha256"]
            )
        args = method.get("factory_args", {})
        required = {
            "template",
            "template-sha256",
            "development-run",
            "profile-receipt",
            "profile-receipt-sha256",
            "auditor-receipt",
            "auditor-receipt-sha256",
            "mc-completion-barrier",
            "mc-completion-barrier-sha256",
            "weights",
            "training-book",
        } | {"book-" + str(seed) for seed in SEEDS[slot]}
        if slot == 5:
            required |= {"books-provenance", "books-provenance-sha256"}
        if "frozen_registration_directory" in method:
            assert not args, (
                "Do not rerun a frozen factory or reset its earliest registration clock"
            )
        else:
            assert set(args) == required
        if "frozen_registration_directory" not in method:

            def original(path):
                matches = [
                    asset
                    for asset in config["assets"]
                    if stage / asset["destination"] == Path(path)
                ]
                return Path(matches[0]["source"]) if matches else Path(path)

            for key in (
                "template",
                "profile-receipt",
                "auditor-receipt",
                "mc-completion-barrier",
            ):
                assert sha(original(args[key])) == args[key + "-sha256"]
            if slot == 5:
                assert (
                    sha(original(args["books-provenance"]))
                    == args["books-provenance-sha256"]
                )
            profile = json.loads(original(args["profile-receipt"]).read_text())
            audit = json.loads(original(args["auditor-receipt"]).read_text())
            assert profile["source_commit"] == audit["source_commit"] == SOURCES[slot]
            assert profile["finished_epoch"] <= profile["absolute_deadline_epoch"]
            assert audit["finished_epoch"] <= audit["absolute_deadline_epoch"]
            assert audit["audit_report"]["epoch"] == 1
            assert audit["audit_report"]["optimizer_committed"] > 0
            expected_profile = (
                "pass-one-development-epoch-and-readonly-audit"
                if slot == 4
                else "pass-one-search-acting-v2-development-epoch-and-fullchronological-audit"
            )
            assert profile["status"] == expected_profile
            assert audit["new_selfplay_transitions_generated"] == 0
            assert audit["optimizer_updates_performed_by_qualification"] == 0
            if slot == 5:
                assert audit["qualified_production_core_sha256"] == sha(
                    original(str(helper(stage, method, "audit_core")))
                )
                assert audit["qualified_report_guard_sha256"] == sha(
                    original(str(helper(stage, method, "adapter_controls")))
                )
        assert set(method["paths_config"]["seeds"]) == {
            str(seed) for seed in SEEDS[slot]
        }
        for row in method["paths_config"]["seeds"].values():
            for field in (
                "run",
                "training_result",
                "audit_controller_result",
                "full_audit",
                "baseline_root",
                "book",
            ):
                assert Path(row[field]).is_absolute()
            assert not Path(row["run"]).parent.exists()
            assert not Path(row["audit_controller_result"]).parent.exists()
            assert not Path(row["baseline_root"]).exists()
        for suffix in (
            "full_audit",
            "audit_controller",
            "posttraining",
            "baseline",
            "fresh_cli_replay",
            "eligibility",
            "parity",
            "latency",
            "final_arms",
            "strength_analysis",
            "all_gates",
        ):
            assert (
                str(
                    Path(method["helper_subdir"])
                    / (PREFIX[slot] + "_" + suffix + ".py")
                )
                in destinations
            )
        assert str(Path(method["helper_subdir"]) / "own45_cohort.py") in destinations
    assert totals[0] == totals[1] == 22620
    assert config["cohort_completion_offset_seconds"] == 14520
    assert now + config["cohort_completion_offset_seconds"] + 7300 + 540 < END


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
    preflight(config, time.time())
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
    first = (
        time.time()
    )  # ONCE, before stage/config/registration writes; all owners share it.
    for method in config["methods"]:
        assert (
            first
            + method["whole_audit_seconds"]
            + method["posttraining_reserve_seconds"]
            < END
        )
    stage = Path(config["stage"])
    stage.mkdir(parents=True, exist_ok=False)
    clock = stage / "common-original-firstclock.json"
    publish(
        clock,
        {
            "schema": "own45-common-original-firstclock-v1",
            "slots": [4, 5],
            "original_training_started_epoch": first,
            "sampler_sha256": sha(__file__),
            "scope": "includes all following stage/config/registration writes",
            "pid": os.getpid(),
        },
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
                "PYTHONPATH": str(Path(repo) / "src"),
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
        for asset in config["assets"]:
            destination = stage / asset["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (
                Path(asset["source"]).open("rb") as incoming,
                destination.open("xb") as outgoing,
            ):
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            assert sha(destination) == asset["sha256"]
        for method in config["methods"]:
            if "frozen_registration_directory" in method:
                destination = stage / ("registration-" + str(method["slot"]))
                destination.mkdir(exist_ok=True)
                original = Path(method["frozen_registration_directory"])
                for name, digest in method["frozen_registration_sha256"].items():
                    if (destination / name).exists():
                        assert sha(destination / name) == digest
                        continue
                    with (
                        (original / name).open("rb") as incoming,
                        (destination / name).open("xb") as out,
                    ):
                        shutil.copyfileobj(incoming, out)
                        out.flush()
                        os.fsync(out.fileno())
                    assert sha(destination / name) == digest
                registration = json.loads(
                    (destination / "registration.json").read_text()
                )
                assert first >= registration["earliest_training_epoch"]
            else:
                run_config(
                    "factory-" + str(method["slot"]),
                    factory_command(config["python"], stage, method, first),
                    method["repo"],
                )
        cohort = {
            "schema": "prospective-own45-completion-cohort-v1",
            "status": "frozen-before-both-formal-executions",
            "slots": [4, 5],
            "latency_order": [4, 5],
            "completion_deadline_epoch": first + 14520,
            "members": [],
        }
        for method in config["methods"]:
            slot = method["slot"]
            root = method["paths_config"]["posttraining"]["root"]
            cohort["members"].append(
                {
                    "slot": slot,
                    "source_commit": SOURCES[slot],
                    "root": root,
                    "coordinator_sha256": sha(helper(stage, method, "posttraining")),
                    "required_process_receipt_names": [
                        *[
                            "replay-" + str(seed) + "-process-result.json"
                            for seed in SEEDS[slot]
                        ],
                        "eligibility-process-result.json",
                        "cuda-parity-process-result.json",
                    ],
                }
            )
        cohort_path = stage / "cohort.json"
        publish(cohort_path, cohort)
        baseline_first = (
            time.time()
        )  # Both baseline ceilings include owner-config writes.
        baseline_deadline = baseline_first + 3600
        assert baseline_deadline < END
        for method in config["methods"]:
            slot = method["slot"]
            registration = stage / ("registration-" + str(slot))
            paths = method["paths_config"]
            paths.update(
                common_original_firstclock_receipt=str(clock),
                common_original_firstclock_receipt_sha256=sha(clock),
                repo=method["repo"],
                python=config["python"],
            )
            paths["posttraining"]["expected_A100_runtime"] = expected
            paths["posttraining"]["cohort"] = {
                "manifest": str(cohort_path),
                "manifest_sha256": sha(cohort_path),
                "helper_sha256": sha(
                    stage / method["helper_subdir"] / "own45_cohort.py"
                ),
            }
            for row in paths["seeds"].values():
                row["original_training_started_epoch"] = first
                row["baseline_deadline_epoch"] = baseline_deadline
            path = stage / ("paths-config-" + str(slot) + ".json")
            publish(path, paths)
            command = [
                config["python"],
                helper(stage, method, "owner_config_factory"),
                "--registration",
                registration / "registration.json",
                "--registration-sha256",
                sha(registration / "registration.json"),
                "--three-input-manifest",
                registration / "three-input-manifest.json",
                "--three-input-manifest-sha256",
                sha(registration / "three-input-manifest.json"),
                "--qualification-config",
                registration / "strength-bindings.json",
                "--qualification-config-sha256",
                sha(registration / "strength-bindings.json"),
                "--paths-config",
                path,
                "--paths-config-sha256",
                sha(path),
                "--helpers",
                stage / method["helper_subdir"],
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
            registration = stage / ("registration-" + str(slot))
            owners = stage / ("owners-" + str(slot))
            bindings = registration / "strength-bindings.json"
            for seed in SEEDS[slot]:
                row = method["paths_config"]["seeds"][str(seed)]
                train_root = Path(row["run"]).parent
                command = [
                    config["python"],
                    helper(stage, method, "training_controller"),
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
                    helper(stage, method, "audit_controller"),
                    "--manifest",
                    manifest,
                    "--manifest-sha256",
                    sha(manifest),
                    "--repo",
                    method["repo"],
                    "--python",
                    config["python"],
                    "--audit-script",
                    helper(stage, method, "full_audit"),
                    "--audit-script-sha256",
                    sha(helper(stage, method, "full_audit")),
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
                    helper(stage, method, "baseline"),
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
                    helper(stage, method, "posttraining"),
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
