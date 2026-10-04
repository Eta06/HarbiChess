"""Fixed final two CPU arms; reuse immutable baseline, no repeated baseline."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

SF_SHA = "0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19"
HARD_DEADLINE = (
    __import__("datetime")
    .datetime(2026, 10, 5, 6, tzinfo=__import__("datetime").timezone.utc)
    .timestamp()
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(p, value):
    with p.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def command(python, candidate, opponent, book, stockfish, seed, directory, seconds):
    return [
        str(python),
        "-m",
        "harbichess.evaluation.portable_arena",
        str(candidate),
        str(opponent),
        "--stockfish",
        str(stockfish),
        "--simulations",
        "16",
        "--nodes",
        "512",
        "--max-plies",
        "400",
        "--wall-seconds",
        str(seconds),
        "--opening-pairs",
        "48",
        "--openings",
        str(book),
        "--split",
        "arena",
        "--seed",
        str(seed),
        "--threads",
        "1",
        "--candidate-root-actions",
        "4",
        "--opponent-root-actions",
        "4",
        "--record-engine-nodes",
        "--progress",
        str(directory / "moves.jsonl"),
        "--output",
        str(directory / "arena.json"),
    ]


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    for name in ("candidate", "initial", "book", "stockfish", "run-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-arena", type=Path, required=True)
    p.add_argument("--baseline-arena-sha256", required=True)
    p.add_argument("--eligible-receipt", type=Path, required=True)
    p.add_argument("--eligible-receipt-sha256", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--seed", type=int, choices=(20261205, 20261206), required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--arm-wall-seconds", type=float, default=3600)
    args = p.parse_args()
    deadline = min(args.deadline_epoch, HARD_DEADLINE)
    if args.arm_wall_seconds != 3600 or deadline <= time.time():
        raise ValueError("frozen3600arm budget and unexpired harddeadline required")
    if sha(args.stockfish) != SF_SHA:
        raise ValueError("frozenSF binary hash mismatch")
    if (
        sha(args.initial)
        != "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    ):
        raise ValueError("initiale8 hash mismatch")
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
    ).strip()
    if source != args.source_commit:
        raise ValueError("sourcecommit mismatch")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=args.repo, text=True
    ).strip():
        raise ValueError("exact clean source required")
    if (
        sha(args.baseline_arena) != args.baseline_arena_sha256
        or sha(args.eligible_receipt) != args.eligible_receipt_sha256
    ):
        raise ValueError("immutable baseline/eligibility receipt mismatch")
    baseline = json.loads(args.baseline_arena.read_text())
    eligible = json.loads(args.eligible_receipt.read_text())
    assert (
        baseline["source_commit"] == source
        and baseline["seed"] == args.seed
        and baseline["candidate_sha256"] == sha(args.initial)
        and baseline["opening_source_sha256"] == sha(args.book)
        and len(baseline["games"]) == 96
    )
    assert (
        eligible["status"]
        == "eligible-both-fixedCURRENT40-for-preregistered-strength-only"
    )
    assert any(
        row["seed"] == args.seed
        and row["epoch"] == 40
        and row["source_commit"] == source
        and row["candidate_sha256"] == sha(args.candidate)
        for row in eligible["seeds"]
    )
    book = json.loads(args.book.read_text())
    roots = book["splits"]["arena"]
    if len(roots) != 48 or len({r["source_game"] for r in roots}) != 48:
        raise ValueError("immutable48rootbook needed")
    args.run_dir.mkdir(exist_ok=False, parents=True)
    inputs = {
        k: {"path": str(getattr(args, k)), "sha256": sha(getattr(args, k))}
        for k in ("candidate", "initial", "book", "stockfish")
    }
    profile = {
        "schema": 1,
        "seed": args.seed,
        "source_commit": source,
        "inputs": inputs,
        "hard_deadline_epoch": deadline,
        "arm_whole_seconds": 3600,
        "baseline_arena_sha256": args.baseline_arena_sha256,
        "eligible_receipt_sha256": args.eligible_receipt_sha256,
        "controller_sha256": sha(__file__),
        "scope": "Sequential CPU strength arms using immutable final model snapshots; no training or teacher labels.",
    }
    publish(args.run_dir / "profile.json", profile)
    receipts = []
    status = "incomplete"

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"owned finalcontroller signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for name, candidate, opponent in [
            ("direct", args.candidate, args.initial),
            ("final_sf", args.candidate, "stockfish"),
        ]:
            for k, v in inputs.items():
                if sha(v["path"]) != v["sha256"]:
                    raise ValueError("immutable input changed")
            directory = args.run_dir / name
            directory.mkdir()
            started = time.time()
            arm_deadline = min(deadline, started + 3600)
            # Internal budget is secondary; parent includes import/setup/teardown/publication.
            cmd = command(
                args.python,
                candidate,
                opponent,
                args.book,
                args.stockfish,
                args.seed,
                directory,
                3600,
            )
            env = {
                **os.environ,
                "PYTHONPATH": str(args.repo / "src"),
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            receipt = {
                "command": cmd,
                "cwd": str(args.repo),
                "started_epoch": started,
                "deadline_epoch": arm_deadline,
                "environment_overrides": {
                    k: env[k]
                    for k in (
                        "PYTHONPATH",
                        "CUDA_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
            }
            publish(directory / "command.json", receipt)
            failure = None
            with (
                (directory / "stdout.log").open("x") as out,
                (directory / "stderr.log").open("x") as err,
            ):
                process = subprocess.Popen(
                    cmd,
                    cwd=args.repo,
                    env=env,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                while process.poll() is None:
                    mempath = Path("/sys/fs/cgroup/memory.current")
                    memory = int(mempath.read_text()) if mempath.exists() else None
                    if time.time() >= arm_deadline:
                        failure = "absolute whole-wall deadline exhausted"
                    elif memory is not None and memory > 15 * 1024**3:
                        failure = "cgroup memory current>15GiB"
                    elif shutil.disk_usage(args.run_dir).free < 8 * 1024**3:
                        failure = "diskfree<8GiB"
                    if failure:
                        terminate(process)
                        break
                    time.sleep(0.5)
            receipt.update(
                finished_epoch=time.time(),
                returncode=process.returncode,
                failure=failure,
                whole_seconds=time.time() - started,
            )
            if receipt["finished_epoch"] > arm_deadline and failure is None:
                receipt["failure"] = "publication after wholedeadline"
            publish(directory / "receipt.json", receipt)
            receipts.append(receipt)
            if receipt["failure"] or process.returncode:
                raise RuntimeError("failed/incomplete" + name)
            result = json.loads((directory / "arena.json").read_text())
            if len(result["games"]) != 96:
                raise ValueError("incomplete96gamearm")
            if (
                result["source_commit"] != source
                or result["opening_source_sha256"] != inputs["book"]["sha256"]
            ):
                raise ValueError("source/bookreceipt mismatch")
        status = "completed-arms-awaiting-independent-analysis-latency-integrity"
    finally:
        if "process" in locals() and process.poll() is None:
            terminate(process)
        publish(
            args.run_dir / "result.json",
            {
                "status": status,
                "receipts": receipts,
                "finished_epoch": time.time(),
                "profile_sha256": sha(args.run_dir / "profile.json"),
            },
        )


if __name__ == "__main__":
    main()
