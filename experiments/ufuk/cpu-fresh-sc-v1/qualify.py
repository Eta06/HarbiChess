"""Bounded whole/pause/resume native proof; not a real-data strength run."""

import argparse
import hashlib
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import torch


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(
                a.contiguous().reshape(-1).view(torch.uint8),
                b.contiguous().reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(equal(x, y) for x, y in zip(a, b, strict=True))
        )
    if isinstance(a, float):
        return type(b) is float and struct.pack("!d", a) == struct.pack("!d", b)
    return type(a) is type(b) and a == b


def run_owned(command, deadline, stdout, stderr, cwd, env):
    if time.time() >= deadline:
        raise TimeoutError("fixed original native-proof deadline exhausted")
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True
    )
    try:
        while process.poll() is None:
            if time.time() >= deadline:
                raise TimeoutError("fixed original native-proof deadline exhausted")
            time.sleep(min(0.2, max(0.01, deadline - time.time())))
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkout",
        "output",
        "weights",
        "protocol",
        "journal",
        "actor-config",
        "journal-helper",
        "feature-helper",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if (
        args.deadline_epoch != protocol["deadline_epoch_by_seed"][str(args.seed)]
        or args.first_epoch >= args.deadline_epoch
        or args.deadline_epoch - args.first_epoch < 60
        or sha(Path(__file__).with_name("train.py")) != protocol["trainer_sha256"]
    ):
        raise ValueError("qualification clock or trainer pin differs from registered proof")
    args.output.mkdir(exist_ok=False)
    shared = [
        sys.executable,
        str(Path(__file__).with_name("train.py")),
        "--weights",
        str(args.weights),
        "--protocol",
        str(args.protocol),
        "--journal",
        str(args.journal),
        "--actor-config",
        str(args.actor_config),
        "--journal-helper",
        str(args.journal_helper),
        "--feature-helper",
        str(args.feature_helper),
        "--source-commit",
        args.source_commit,
        "--seed",
        str(args.seed),
        "--deadline-epoch",
        str(args.deadline_epoch),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(args.checkout / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    result = {
        "schema": "fresh-qsearch-additive-SC-native-restart-proof-v1",
        "status": "failed-preserved",
        "source_commit": args.source_commit,
        "seed": args.seed,
        "first_epoch": args.first_epoch,
        "original_deadline_epoch": args.deadline_epoch,
        "journal_sha256": sha(args.journal),
        "actor_config_sha256": sha(args.actor_config),
        "sc_helper_sha256": protocol["sc_helper_sha256"],
        "parent_mc_trainer_sha256": protocol["parent_mc_trainer_sha256"],
        "scope": (
            "8-update whole/pause/resume and strict fresh-process native-load proof; "
            "no strength claim"
        ),
    }
    try:
        invocations = (
            ("whole", "whole", None, 8),
            ("pause", "split", None, 4),
            ("resume", "split", "step-00000004", 8),
        )
        for name, output_name, resume_tag, stop_at in invocations:
            command = [
                *shared,
                "--output",
                str(args.output / output_name),
                "--stop-at",
                str(stop_at),
            ]
            if resume_tag:
                command += ["--resume", str(args.output / output_name / "checkpoints" / resume_tag)]
            with (
                (args.output / f"{name}.stdout.log").open("xb") as out,
                (args.output / f"{name}.stderr.log").open("xb") as err,
            ):
                run_owned(command, args.deadline_epoch, out, err, args.checkout, env)
        snapshots = []
        for name in ("whole", "split"):
            checkpoint = args.output / name / "checkpoints/step-00000008"
            manifest = json.loads((checkpoint / "checkpoint.json").read_text())
            assert manifest["schema"] == "fresh-qsearch-additive-own-sc-native-v1"
            assert set(manifest["artifacts"]) == {"training.pt"}
            assert sha(checkpoint / "training.pt") == manifest["artifacts"]["training.pt"]
            snapshots.append(
                torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=True)
            )
        assert equal(*snapshots), "head/Adam/all RNG/data bindings differ after resume"
        assert snapshots[0]["accepted"] == snapshots[0]["attempted"] == 8
        for name in ("whole", "split"):
            for step in (0, 4, 8):
                command = [
                    *shared,
                    "--output",
                    str(args.output / name),
                    "--resume",
                    str(args.output / name / f"checkpoints/step-{step:08d}"),
                    "--audit-only",
                ]
                with (
                    (args.output / f"strict-{name}-{step}.stdout.log").open("xb") as out,
                    (args.output / f"strict-{name}-{step}.stderr.log").open("xb") as err,
                ):
                    run_owned(command, args.deadline_epoch, out, err, args.checkout, env)
        result.update(
            status="PASS-native-restart-not-strength",
            exact_whole_vs_split_native_payload=True,
            all6_full_native_fresh_process_loads=True,
            accepted_updates=8,
            dataset_contract_sha256=snapshots[0]["contract"]["dataset_sha256"],
        )
    except Exception as exc:
        result["error"] = repr(exc)
    result["finished_epoch"] = time.time()
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "PASS-native-restart-not-strength":
        raise RuntimeError(result.get("error", "native restart proof failed"))


if __name__ == "__main__":
    main()
