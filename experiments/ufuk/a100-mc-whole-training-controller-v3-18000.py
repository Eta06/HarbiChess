"""Explicit root-invoked method2 fixed40 pause1/freshresume40 controller.

Requires finalized prospective registration+immutable SHA manifest. No remote
access, automatic retry, source repair, budget reset or alternate candidate.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path

HARD_DEADLINE = 1791170400.0
SEEDS = (20261205, 20261206)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def publish(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def install_interrupt_handlers():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"owned controller signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)


def cpu_total_memory():
    path = Path("/sys/fs/cgroup/memory.current")
    if path.exists():
        return int(path.read_text()), "cgroup.memory.current"
    values = {
        line.split(":")[0]: int(line.split(":")[1].strip().split()[0]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line
    }
    return values["MemTotal"] - values[
        "MemAvailable"
    ], "system-used-MemTotal-minus-MemAvailable"


def verify_boundary(root, seed, source, epoch, deadline):
    run = root / "run"
    manifest = read(run / "checkpoints" / f"epoch-{epoch:08d}" / "checkpoint.json")
    state = read(run / "checkpoints" / f"epoch-{epoch:08d}" / "actor.json")
    metadata = read(run / "metadata.json")
    assert (
        metadata["source_commit"] == source
        and metadata["config"]["seed"] == seed
        and metadata["absolute_deadline_epoch"] == deadline
        and metadata["max_epochs"] == 40
        and metadata["checkpoint_interval"] == 1
    )
    assert manifest["run_config"]["config"] == metadata["config"]
    assert manifest["run_config"]["input_sha256"] == {
        name: row["sha256"] for name, row in metadata["inputs"].items()
    }
    assert (
        manifest["schema"] == "torch-fullgame-native-cuda-v1"
        and manifest["source_commit"] == source
        and manifest["state"] == state
    )
    assert (
        manifest["run_config"]["config"]["seed"] == seed
        and state["epoch"] == epoch
        and state["actor_steps"] == epoch * 256
        and state["fresh_transitions"] == epoch * 32768
    )
    assert state["replay_buffer"] == "closed-empty" and state["ppo_pass"] == "closed"
    for name, digest in manifest["artifacts"].items():
        assert sha(run / "checkpoints" / f"epoch-{epoch:08d}" / name) == digest
    assert (run / "journal" / f"epoch-{epoch:08d}.json.gz").is_file()
    progress = read(run / "progress.json")
    assert (
        progress["status"] == "completed"
        and progress["epoch"] == epoch
        and progress["actor_steps"] == epoch * 256
        and progress["closed_boundary"] is True
        and progress["absolute_deadline_epoch"] == deadline
        and progress["finished_epoch"] <= deadline
    )
    assert progress["source_commit"] == source
    for name in (
        "optimizer_accepted_updates",
        "optimizer_attempted_updates",
        "optimizer_rejected_updates",
        "sample_chain_sha256",
    ):
        assert progress[name] == state[name]
    return state


def main():
    p = argparse.ArgumentParser()
    for name in ("repo", "python", "registration", "input-manifest", "root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--started-epoch", type=float, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--registration-sha256", required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    a = p.parse_args()
    assert sha(a.registration) == a.registration_sha256
    registration = read(a.registration)
    assert registration["status"] == "frozen-before-formal-execution"
    source = registration["source_commit"]
    assert len(source) == 40 and all(c in "0123456789abcdef" for c in source)
    assert (
        registration["qualification_ledger_slot"] == 2
        and registration["seeds"] == list(SEEDS)
        and registration["proposed_epochs"] == 40
        and registration["whole_training_seconds_per_seed"] == 18000
    )
    assert registration["controller_sha256"] == sha(Path(__file__))
    assert (
        registration["memory_max_bytes"] == 64 * 1024**3
        and registration["disk_min_free_bytes"] == 8 * 1024**3
    )
    repo = a.repo.resolve()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        == source
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    inputs = read(a.input_manifest)
    assert registration["immutable_inputs"]["manifest_sha256"] == sha(a.input_manifest)
    seedinputs = inputs["seeds"][str(a.seed)]
    paths = {name: Path(row["path"]).resolve() for name, row in seedinputs.items()}
    assert set(paths) == {"initial_weights", "book", "experiment_config"}
    seedinputs = dict(seedinputs)
    seedinputs["protocol"] = {
        "path": str(a.registration.resolve()),
        "sha256": a.registration_sha256,
    }
    paths["protocol"] = a.registration.resolve()
    for name, path in paths.items():
        assert sha(path) == seedinputs[name]["sha256"]
    assert (
        sha(paths["initial_weights"])
        == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    )
    assert (
        paths["protocol"] == a.registration.resolve()
        and seedinputs["protocol"]["sha256"] == a.registration_sha256
    )
    config = read(paths["experiment_config"])
    assert config["seed"] == a.seed
    for name in (
        "actors",
        "objective",
        "schedule",
        "epoch_steps",
        "learning_rate",
        "weight_decay",
        "device",
    ):
        assert config[name] == registration[name]
    assert (
        config["actors"]["games"] == 128
        and config["actors"]["max_additional_plies"] == 256
        and config["actors"]["temperature"] == 1
        and config["epoch_steps"] == 256
        and config["device"] == "cuda:0"
    )
    started = a.started_epoch
    deadline = a.deadline_epoch
    assert deadline == started + 18000 and started <= time.time() < deadline
    assert deadline < HARD_DEADLINE, (
        "Insufficient physical time for full18000budget; no silent shortening"
    )
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    profile = {
        "schema": "method2-wholetraining-controller-v1",
        "seed": a.seed,
        "source_commit": source,
        "started_epoch": started,
        "original_absolute_deadline_epoch": deadline,
        "registered_whole_seconds": 18000,
        "hard_user_deadline_epoch": HARD_DEADLINE,
        "registration_sha256": a.registration_sha256,
        "input_manifest_sha256": sha(a.input_manifest),
        "controller_sha256": sha(Path(__file__)),
        "inputs": seedinputs,
        "fixed_candidate": "CURRENTepoch40",
        "scope": "Training completion only; no strength/promotion.",
    }
    publish(root / "profile.json", profile)
    env = {
        **os.environ,
        "PYTHONPATH": str(repo / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    common = [
        str(a.python),
        "-m",
        "harbichess.training.torch_fullgame_run",
        str(root / "run"),
        "--weights",
        str(paths["initial_weights"]),
        "--book",
        str(paths["book"]),
        "--config",
        str(paths["experiment_config"]),
        "--protocol",
        str(paths["protocol"]),
        "--source-commit",
        source,
        "--max-epochs",
        "40",
        "--checkpoint-interval",
        "1",
        "--deadline-epoch",
        str(deadline),
        "--memory-max-bytes",
        str(64 * 1024**3),
        "--disk-min-free-bytes",
        str(8 * 1024**3),
    ]
    phases = (
        ("pause1", common + ["--stop-at", "1"]),
        (
            "freshresume40",
            common + ["--resume", str(root / "run/checkpoints/epoch-00000001")],
        ),
    )
    receipts = []
    status = "failed-or-incomplete-fixed40"
    error = None
    final = None
    install_interrupt_handlers()
    try:
        for name, args in phases:
            for key, path in paths.items():
                assert sha(path) == seedinputs[key]["sha256"]
            begin = time.time()
            receipt = {
                "phase": name,
                "command": args,
                "started_epoch": begin,
                "same_original_deadline_epoch": deadline,
            }
            publish(root / (name + "-command.json"), receipt)
            if begin >= deadline:
                raise TimeoutError(
                    "Original wholetraining budget exhausted before next phase"
                )
            process = None
            failure = None
            try:
                with (
                    (root / (name + ".stdout.log")).open("x") as out,
                    (root / (name + ".stderr.log")).open("x") as err,
                ):
                    process = subprocess.Popen(
                        args,
                        cwd=repo,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                    )
                    publish(
                        root / (name + "-owner.json"),
                        {
                            "pid": process.pid,
                            "owned_process_group": process.pid,
                            "started_epoch": begin,
                        },
                    )
                    while process.poll() is None:
                        memory, metric = cpu_total_memory()
                        if time.time() >= deadline:
                            failure = "original wholedeadline exhausted"
                        elif memory > 64 * 1024**3:
                            failure = "registered64GiB memory ceiling exceeded"
                        elif shutil.disk_usage(root).free < 8 * 1024**3:
                            failure = "registered8GiB disk free floor violated"
                        if failure:
                            terminate(process)
                            break
                        time.sleep(0.5)
                if process.returncode != 0:
                    failure = failure or f"childreturncode{process.returncode}"
                if time.time() > deadline:
                    failure = (
                        failure or "child/publication after original wholedeadline"
                    )
            except BaseException as e:
                failure = repr(e)
                if process is not None and process.poll() is None:
                    terminate(process)
                raise
            finally:
                receipt.update(
                    finished_epoch=time.time(),
                    returncode=process.returncode if process else None,
                    failure=failure,
                    whole_phase_seconds=time.time() - begin,
                )
                receipts.append(receipt)
                publish(root / (name + "-receipt.json"), receipt)
            if failure:
                raise RuntimeError(failure)
            verify_boundary(
                root, a.seed, source, 1 if name == "pause1" else 40, deadline
            )
        assert sorted(
            p.name
            for p in (root / "run/checkpoints").iterdir()
            if p.is_dir() and p.name.startswith("epoch-")
        ) == [f"epoch-{i:08d}" for i in range(41)]
        assert sorted(
            p.name for p in (root / "run/journal").glob("epoch-*.json.gz")
        ) == [f"epoch-{i:08d}.json.gz" for i in range(1, 41)]
        final = root / "run/checkpoints/epoch-00000040/model.safetensors"
        if time.time() > deadline:
            raise TimeoutError(
                "Final inventory publication after original wholedeadline"
            )
        status = "completed-fixedCURRENT40-awaiting-independent-integrity-latency-portability-strength"
    except BaseException as e:
        error = {"error": repr(e), "traceback": traceback.format_exc()}
        raise
    finally:
        result = {
            "status": status,
            "profile": profile,
            "receipts": receipts,
            "error": error,
            "finished_epoch": time.time(),
            "whole_seconds_since_first_launch": time.time() - started,
            "fixed_current40_sha256": sha(final)
            if status.startswith("completed")
            else None,
            "scope": "No checkpointselection or qualification; preserve failed/incomplete runs and all native/journal evidence.",
        }
        publish(root / "result.json", result)


if __name__ == "__main__":
    main()
