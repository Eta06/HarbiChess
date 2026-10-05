"""Prospective root-invoked SEARCH-ACTING-v2 fixed-epoch pause/resume; never invokes MC schemas."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

SEEDS = (20261825, 20261826)
END = 1791180000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_mc_barrier(barrier, started):
    """Completion receipts only; never inspect heldout arena scores or latency ratios."""
    expected = {
        "replay-20261205-process-result.json",
        "replay-20261206-process-result.json",
        "eligibility-process-result.json",
        "cuda-parity-process-result.json",
        "latency-process-result.json",
    }
    receipts = barrier["process_receipts"]
    assert {Path(row["path"]).name for row in receipts} == expected
    assert len(receipts) == len(expected)
    assert barrier["coordinator_sha256"] == (
        "686265b09284d1e8c879406510910991d2d8f1a71bf60b610372eac00e76e100"
    )
    for row in receipts:
        assert sha(row["path"]) == row["sha256"]
        result = read(row["path"])
        assert result["returncode"] == 0
        assert result["finished_epoch"] <= result["deadline_epoch"]
        assert result["finished_epoch"] <= started
    # Bind the completed data file without reading speed gates or model outcomes.
    latency = barrier["latency_receipt"]
    assert Path(latency["path"]).name == "latency.json"
    assert sha(latency["path"]) == latency["sha256"]
    assert len({Path(row["path"]).parent.resolve() for row in receipts}) == 1
    assert Path(latency["path"]).parent.resolve() == Path(receipts[0]["path"]).parent.resolve()


def validate_native(run, epoch, source):
    directory = run / "checkpoints" / f"epoch-{epoch:08d}"
    manifest = read(directory / "checkpoint.json")
    state = read(directory / "actor.json")
    assert manifest["schema"] == "torch-search-acting-native-cuda-v3"
    assert manifest["run_config"]["schema"] == "torch-fresh-sparse-search-acting-v3"
    assert manifest["source_commit"] == source and manifest["state"] == state
    assert state["epoch"] == epoch and state["actor_steps"] == epoch * 256
    assert state["fresh_transitions"] == epoch * 32768
    assert state["pending_search_schedule"] == state["replay_buffer"] == "closed-empty"
    assert state["training_pass"] == "closed" and len(state["search_rngs"]) == 128
    assert set(manifest["artifacts"]) == {
        "model.safetensors",
        "base.safetensors",
        "behavior.safetensors",
        "training.pt",
        "actor.json",
        "last-frozen-epoch.json.gz",
    }
    for filename, digest in manifest["artifacts"].items():
        assert sha(directory / filename) == digest
    return manifest


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


def validate_training_clock(registration, first, deadline, now):
    from own8_schedule_v3 import validate_clock

    validate_clock(registration, first, deadline, now)


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "python", "registration", "input-manifest", "root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--started-epoch", type=float, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    assert sha(args.registration) == args.registration_sha256
    reg = read(args.registration)
    assert (
        reg["status"] == "frozen-before-formal-execution" and reg["qualification_ledger_slot"] == 8
    )
    assert (
        reg["seeds"] == list(SEEDS)
        and reg["learner_schema"] == "torch-fresh-sparse-search-acting-v3"
    )
    assert reg["controller_sha256"] == sha(Path(__file__))
    for name, digest in reg["helper_sha256"].items():
        assert sha(Path(__file__).with_name(name)) == digest
    assert reg["infrastructure_profile_pass"] is True
    epochs = reg["fixed_epochs"]
    assert type(epochs) is int and epochs == 8
    deadline = args.deadline_epoch
    assert deadline == args.started_epoch + reg["whole_training_seconds_per_seed"]
    validate_training_clock(reg, args.started_epoch, deadline, time.time())
    validate_mc_barrier(reg["mc_completion_barrier"], args.started_epoch)
    from own8_schedule_v3 import verify_previous

    verify_previous(reg["terminal45_barrier_binding"], sha)
    source = reg["source_commit"]
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
        == source
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=args.repo, text=True
    ).strip()
    assert sha(args.input_manifest) == reg["three_input_manifest_sha256"]
    inputs = read(args.input_manifest)["seeds"][str(args.seed)]
    assert set(inputs) == {"initial_weights", "book", "experiment_config"}
    inputs = {
        **inputs,
        "protocol": {
            "path": str(args.registration.resolve()),
            "sha256": args.registration_sha256,
        },
    }
    for info in inputs.values():
        assert sha(info["path"]) == info["sha256"]
    config = read(inputs["experiment_config"]["path"])
    assert config == reg["configs"][str(args.seed)]
    assert (
        config["seed"] == args.seed
        and config["actors"]["games"] == 128
        and config["epoch_steps"] == 256
    )
    assert config["actors"]["temperature"] == 1 and config["device"] == "cuda:0"
    assert (
        inputs["initial_weights"]["sha256"]
        == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    )
    args.root.mkdir(parents=True, exist_ok=False)
    publish(
        args.root / "original-launch.json",
        {
            "source_commit": source,
            "seed": args.seed,
            "started_epoch": args.started_epoch,
            "absolute_deadline_epoch": deadline,
            "fixed_epochs": epochs,
            "protocol_sha256": args.registration_sha256,
            "inputs": inputs,
        },
    )
    process = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Owned process interruption {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    env = {
        **os.environ,
        "PYTHONPATH": str(args.repo.resolve() / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    common = [
        str(args.python),
        "-m",
        "harbichess.training.torch_search_acting_run",
        str(args.root / "run"),
        "--weights",
        inputs["initial_weights"]["path"],
        "--book",
        inputs["book"]["path"],
        "--config",
        inputs["experiment_config"]["path"],
        "--protocol",
        str(args.registration.resolve()),
        "--source-commit",
        source,
        "--max-epochs",
        str(epochs),
        "--checkpoint-interval",
        "1",
        "--deadline-epoch",
        str(deadline),
        "--memory-max-bytes",
        str(64 * 1024**3),
        "--disk-min-free-bytes",
        str(8 * 1024**3),
    ]
    status = "failed-or-incomplete-own8-no-retry"
    try:
        for name, command, boundary in [
            ("pause1", [*common, "--stop-at", "1"], 1),
            (
                "freshresumeE",
                [
                    *common,
                    "--resume",
                    str(args.root / "run/checkpoints/epoch-00000001"),
                ],
                epochs,
            ),
        ]:
            publish(
                args.root / (name + "-command.json"),
                {"argv": command, "original_deadline_epoch": deadline},
            )
            with (
                (args.root / (name + ".stdout")).open("x") as out,
                (args.root / (name + ".stderr")).open("x") as err,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=args.repo,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                while process.poll() is None:
                    if time.time() >= deadline:
                        raise TimeoutError("Original SEARCH-ACTING-v2 training ceiling exhausted")
                    time.sleep(0.5)
            assert process.returncode == 0 and time.time() <= deadline
            validate_native(args.root / "run", boundary, source)
        assert sorted(p.name for p in (args.root / "run/checkpoints").glob("epoch-*")) == [
            f"epoch-{i:08d}" for i in range(epochs + 1)
        ]
        status = "completed-fixedSEARCH_ACTINGv2-currentE-awaiting-independent-qualification"
    finally:
        if process is not None:
            stop(process)
        publish(
            args.root / "result.json",
            {
                "status": status,
                "fixed_epochs": epochs,
                "finished_epoch": time.time(),
                "original_deadline_epoch": deadline,
                "source_commit": source,
                "candidate_sha256": sha(
                    args.root / "run/checkpoints" / f"epoch-{epochs:08d}" / "model.safetensors"
                )
                if status.startswith("completed")
                else None,
            },
        )


if __name__ == "__main__":
    main()
