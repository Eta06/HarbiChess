"""Explicit frozen e8-v-SF512 arm only; outcomes cannot steer training."""

import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "training_control",
    Path(__file__).with_name("a100-mc-whole-training-controller-v3-18000.py"),
)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
BOOKS = {
    20261205: "b09f7dc1f4a8510c595c9732df195746f7a7455d1ccdc6ca8046301f70b3b0dc",
    20261206: "b790f63d616b5d54164e6d1358acffd3ae1dcf7f6c66f20865f9f09f0fadf70b",
}


def main():
    p = argparse.ArgumentParser()
    for name in (
        "repo",
        "python",
        "registration",
        "weights",
        "book",
        "stockfish",
        "root",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--registration-sha256", required=True)
    p.add_argument("--seed", type=int, choices=c.SEEDS, required=True)
    p.add_argument("--started-epoch", type=float, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    assert c.sha(a.registration) == a.registration_sha256
    r = c.read(a.registration)
    assert r["status"] == "frozen-before-formal-execution" and r["qualification_ledger_slot"] == 2
    assert (
        a.deadline_epoch == a.started_epoch + 3600
        and a.started_epoch <= time.time() < a.deadline_epoch <= c.HARD_DEADLINE
    )
    assert (
        c.sha(a.weights) == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
        and c.sha(a.book) == BOOKS[a.seed]
    )
    assert c.sha(a.stockfish) == "0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19"
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.repo, text=True).strip()
        == r["source_commit"]
        and not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=a.repo, text=True
        ).strip()
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
        "schema": "method2-frozen-baseline-only-v1",
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
            "Heldout evaluation after formal freeze; never training labels or "
            "candidate/config/epoch selection."
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
        with (
            (a.root / "stdout.log").open("x") as out,
            (a.root / "stderr.log").open("x") as err,
        ):
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
                memory, _ = c.cpu_total_memory()
                if (
                    time.time() >= a.deadline_epoch
                    or memory > 15 * 1024**3
                    or __import__("shutil").disk_usage(a.root).free < 8 * 1024**3
                ):
                    c.terminate(process)
                    raise TimeoutError("Baseline original3600whole/resource guard exhausted")
                time.sleep(0.5)
        assert process.returncode == 0 and time.time() <= a.deadline_epoch
        result = c.read(a.root / "arena.json")
        assert (
            result["source_commit"] == r["source_commit"]
            and result["seed"] == a.seed
            and len(result["games"]) == 96
            and result["max_plies"] == 400
            and result["stockfish_nodes"] == 512
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
            {
                "status": receipt["status"],
                "seed": a.seed,
                "arena_sha256": receipt["arena_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
