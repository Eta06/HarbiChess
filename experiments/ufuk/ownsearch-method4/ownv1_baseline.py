"""Explicit frozen e8-v-SF512 arm only; outcomes cannot steer training."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import ownv1_strength_runtime as c
from ownv1_strength_config import BOOKS, bind_cli, validate_binding

Q = {}


def resource_snapshot(root, deadline):
    memory, metric = c.cpu_total_memory()
    disk_free = __import__("shutil").disk_usage(root).free
    now = time.time()
    violations = []
    if now >= deadline:
        violations.append("absolute_deadline")
    if memory > 64 * 1024**3:
        violations.append("memory_above_64GiB")
    if disk_free < 8 * 1024**3:
        violations.append("disk_free_below_8GiB")
    return {
        "observed_epoch": now,
        "original_deadline_epoch": deadline,
        "memory_metric": metric,
        "memory_bytes": memory,
        "memory_limit_bytes": 64 * 1024**3,
        "disk_free_bytes": disk_free,
        "disk_free_min_bytes": 8 * 1024**3,
        "violations": violations,
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("repo", "python", "registration", "weights", "book", "stockfish", "root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--registration-sha256", required=True)
    p.add_argument("--seed", type=int, choices=c.SEEDS, required=True)
    p.add_argument("--started-epoch", type=float, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    validate_binding(a, Q)
    assert c.sha(a.registration) == a.registration_sha256
    r = c.read(a.registration)
    assert r["status"] == "frozen-before-formal-execution" and r["qualification_ledger_slot"] == 4
    assert (
        a.deadline_epoch == a.started_epoch + 3600
        and a.started_epoch <= time.time() < a.deadline_epoch <= c.HARD_DEADLINE
    )
    assert (
        c.sha(a.weights) == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
        and c.sha(a.book) == BOOKS[a.seed]
    )
    assert c.sha(a.stockfish) == "0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19"
    assert subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=a.repo, text=True
    ).strip() == r["source_commit"] and (
        not subprocess.check_output(["git", "status", "--porcelain"], cwd=a.repo, text=True).strip()
    )
    a.root.mkdir(parents=True, exist_ok=False)
    cmd = [
        str(a.python),
        "-m",
        "harbichess.evaluation.portable_arena",
        str(a.weights.resolve()),
        "stockfish",
        "--stockfish",
        str(a.stockfish.resolve()),
        "--simulations",
        "16",
        "--nodes",
        "512",
        "--max-plies",
        "400",
        "--wall-seconds",
        "3600",
        "--opening-pairs",
        "48",
        "--openings",
        str(a.book.resolve()),
        "--split",
        "arena",
        "--seed",
        str(a.seed),
        "--threads",
        "1",
        "--candidate-root-actions",
        "4",
        "--opponent-root-actions",
        "4",
        "--record-engine-nodes",
        "--progress",
        str((a.root / "moves.jsonl").resolve()),
        "--output",
        str((a.root / "arena.json").resolve()),
    ]
    receipt = {
        "schema": "ownv1-frozen-baseline-only-v1",
        "qualification_ledger_slot": 4,
        "resource_scope": (
            "Prospectively registered A100-host CPU evaluation,64GiB cgroup c"
            "eiling; original formal2 guard failure retained, exact triggerin"
            "g condition unknown"
        ),
        "seed": a.seed,
        "source_commit": r["source_commit"],
        "registration_sha256": a.registration_sha256,
        "weights_sha256": c.sha(a.weights),
        "book_sha256": c.sha(a.book),
        "command": cmd,
        "started_epoch": a.started_epoch,
        "original_deadline_epoch": a.deadline_epoch,
        "status": "running",
        "scope": (
            "Heldout evaluation after formal freeze; never training labels or"
            " candidate/config/epoch selection."
        ),
    }
    c.publish(a.root / "command.json", receipt)
    env = {
        **os.environ,
        "PYTHONPATH": str(a.repo.resolve() / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    process = None
    c.install_interrupt_handlers()
    try:
        with (a.root / "stdout.log").open("x") as out, (a.root / "stderr.log").open("x") as err:
            process = subprocess.Popen(
                cmd,
                cwd=a.repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            while process.poll() is None:
                snapshot = resource_snapshot(a.root, a.deadline_epoch)
                receipt["last_resource_snapshot"] = snapshot
                if snapshot["violations"]:
                    receipt["guard_failure_snapshot"] = snapshot
                    c.publish(a.root / "guard-failure.json", snapshot)
                    c.terminate(process)
                    raise TimeoutError(
                        "Baseline original3600whole/resource guard exhausted: "
                        + ",".join(snapshot["violations"])
                    )
                time.sleep(0.5)
        assert process.returncode == 0 and time.time() <= a.deadline_epoch
        result = c.read(a.root / "arena.json")
        assert (
            result["source_commit"] == r["source_commit"]
            and result["seed"] == a.seed
            and (len(result["games"]) == 96)
            and (result["max_plies"] == 400)
            and (result["stockfish_nodes"] == 512)
        )
        receipt.update(
            status="completed-frozen-baseline-outcomes-withheld-from-training-decisions",
            arena_sha256=c.sha(a.root / "arena.json"),
        )
    except BaseException as e:
        if process is not None and process.poll() is None:
            c.terminate(process)
        receipt.update(status="failed-or-incomplete-baseline-preserved", error=repr(e))
        raise
    finally:
        receipt.update(
            finished_epoch=time.time(),
            whole_seconds=time.time() - a.started_epoch,
            returncode=process.returncode if process else None,
        )
        c.publish(a.root / "result.json", receipt)
    print(
        json.dumps(
            {"status": receipt["status"], "seed": a.seed, "arena_sha256": receipt["arena_sha256"]}
        )
    )


if __name__ == "__main__":
    bind_cli(globals())
    main()
