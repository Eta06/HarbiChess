"""Bounded full-critic native restart proof; no strength or model-selection claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8),
                right.contiguous().reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(exact(left[key], right[key]) for key in left)
        )
    if isinstance(left, list | tuple):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(exact(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def run_owned(command, deadline, stdout, stderr, cwd, env):
    if time.time() >= deadline:
        raise TimeoutError("original fixed qualification deadline exhausted")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    try:
        while process.poll() is None:
            if time.time() >= deadline:
                raise TimeoutError("original fixed qualification deadline exhausted")
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
        "common-trainer",
        "search-helper",
        "array-encoder",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--first-epoch", type=float, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    seed = protocol.get("seed_pairing", {}).get(str(args.seed))
    if (
        seed is None
        or args.deadline_epoch != seed.get("original_deadline_epoch")
        or args.first_epoch >= args.deadline_epoch
        or args.deadline_epoch - args.first_epoch < 60
        or sha(Path(__file__).with_name("train.py")) != protocol.get("trainer_sha256")
        or sha(Path(__file__).with_name("core.py")) != protocol.get("core_sha256")
        or sha(Path(__file__).with_name("native.py")) != protocol.get("native_sha256")
        or sha(Path(__file__)) != protocol.get("qualifier_sha256")
    ):
        raise ValueError("qualification clock or full-critic code pins differ")
    if args.seed not in protocol.get("seeds", []):
        raise ValueError("qualification seed is not registered")
    args.output.mkdir(parents=True, exist_ok=False)
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
        "--common-trainer",
        str(args.common_trainer),
        "--search-helper",
        str(args.search_helper),
        "--array-encoder",
        str(args.array_encoder),
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
        "schema": "fresh-qsearch-full-critic-native-restart-proof-v1",
        "status": "failed-preserved",
        "source_commit": args.source_commit,
        "seed": args.seed,
        "first_epoch": args.first_epoch,
        "original_deadline_epoch": args.deadline_epoch,
        "journal_sha256": sha(args.journal),
        "actor_config_sha256": sha(args.actor_config),
        "trainer_sha256": sha(Path(__file__).with_name("train.py")),
        "scope": (
            "whole8 vs pause4/resume8 exact full-native CPU proof; no real fit or strength claim"
        ),
    }
    try:
        for label, output_name, resume_step, stop in (
            ("whole", "whole", None, 8),
            ("pause", "split", None, 4),
            ("resume", "split", 4, 8),
        ):
            command = [*shared, "--output", str(args.output / output_name), "--stop-at", str(stop)]
            if resume_step is not None:
                command += [
                    "--resume",
                    str(args.output / output_name / "checkpoints" / f"step-{resume_step:08d}"),
                ]
            with (
                (args.output / f"{label}.stdout.log").open("xb") as out,
                (args.output / f"{label}.stderr.log").open("xb") as err,
            ):
                run_owned(command, args.deadline_epoch, out, err, args.checkout, env)
        payloads = []
        for name in ("whole", "split"):
            path = args.output / name / "checkpoints/step-00000008"
            manifest = json.loads((path / "checkpoint.json").read_text())
            if (
                manifest.get("schema") != protocol["native"]["schema"]
                or manifest.get("accepted") != 8
                or sha(path / "training.pt") != manifest.get("training_pt_sha256")
            ):
                raise ValueError("step-8 full-native artifact manifest/SHA differs")
            payloads.append(torch.load(path / "training.pt", map_location="cpu", weights_only=True))
        if not exact(*payloads):
            raise ValueError("whole and paused/resumed full-native payloads differ")
        if payloads[0].get("accepted") != 8:
            raise ValueError("whole/resumed native cursor differs")
        # Strict loads at 0/4/8 in each independent process; a load failure is retained.
        for name in ("whole", "split"):
            for step in (0, 4, 8):
                command = [
                    *shared,
                    "--output",
                    str(args.output / name),
                    "--resume",
                    str(args.output / name / "checkpoints" / f"step-{step:08d}"),
                    "--audit-only",
                ]
                with (
                    (args.output / f"strict-{name}-{step}.stdout.log").open("xb") as out,
                    (args.output / f"strict-{name}-{step}.stderr.log").open("xb") as err,
                ):
                    run_owned(command, args.deadline_epoch, out, err, args.checkout, env)
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("original fixed qualification deadline exhausted before publication")
        result.update(
            status="PASS-native-restart-not-fit",
            exact_whole_vs_pause_resume_native_payload=True,
            all_six_strict_fresh_process_loads=True,
            accepted_updates=8,
            common_dataset_sha256=payloads[0]["contract"]["common_dataset_sha256"],
            search_dataset_sha256=payloads[0]["contract"]["search_extended_dataset_sha256"],
        )
    except Exception as exc:
        result["error"] = repr(exc)
    result["finished_epoch"] = time.time()
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "PASS-native-restart-not-fit":
        raise RuntimeError(result.get("error", "full-critic qualification failed"))


if __name__ == "__main__":
    main()
