"""ROOT immutable real7200 collection registration; one CPU worker per seed."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/workspace/work/harbichess/continuation-20261007")
SOURCE = BASE / "closed-terminal-own-v1/source"
END = 1791448916.685839


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True, choices=[20262905, 20262906])
    p.add_argument("--cpu-core", type=int, required=True)
    args = p.parse_args()
    source_hashes = {str(f): sha(f) for f in sorted(SOURCE.glob("*.py"))}
    old = json.loads((BASE / f"nnue-own-producer-v2/registrations/{args.seed}.json").read_bytes())
    records = BASE / f"closedterminal-collection-{args.seed}"
    records.mkdir(exist_ok=False)
    first = time.time()
    reg = {
        **old,
        "schema": "own-nnue-closed-terminal-collection-registration-v1",
        "original_first_epoch": first,
        "original_deadline_epoch": min(first + 7200, END),
        "operator_end_epoch": END,
        "cpu_core": args.cpu_core,
        "plies_per_root": 400,
        "actor_row_limit": 4096,
        "output_path": f"/dev/shm/harbichess-closedterminal-v1/{args.seed}",
        "producer_source_sha256": {
            f.name: source_hashes[str(f)] for f in sorted(SOURCE.glob("*.py"))
        },
        "ROOT_launcher_sha256": sha(__file__),
        "ROOT_preregistration": {
            "commit": "7b0095c254a71e837cf4030fe8c024ad1132fdc3",
            "path": "docs/runs/UFUK-DEVAM-target-ablation-preregistration-20261007.md",
            "sha256": sha(
                "/workspace/HarbiChess/docs/runs/UFUK-DEVAM-target-ablation-preregistration-20261007.md"
            ),
        },
    }
    write(records / "registration.json", reg)
    command = [
        sys.executable,
        str(SOURCE / "run_collection.py"),
        "--registration",
        str(records / "registration.json"),
    ]
    result = dict(
        status="running",
        first=first,
        deadline=reg["original_deadline_epoch"],
        command=command,
        registration_sha256=sha(records / "registration.json"),
        source_sha256=source_hashes,
        ROOT_pid=os.getpid(),
        completed_teacher_query_count=0,
    )
    child = None
    try:
        if any(sha(f) != h for f, h in source_hashes.items()):
            raise RuntimeError("frozen source closure changed")
        with (records / "stdout.log").open("xb") as out, (records / "stderr.log").open("xb") as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            result["child_pid"] = child.pid
            result["child_startticks"] = (
                Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
            write(records / "owner.json", result)
            child.wait(timeout=max(0.001, reg["original_deadline_epoch"] - time.time()))
        if child.returncode:
            raise RuntimeError("actual collection child failed, no same-phase retry")
        receipt = Path(reg["output_path"]) / "receipt.json"
        packet = json.loads(receipt.read_bytes())
        result.update(
            status=packet["status"],
            receipt={"path": str(receipt), "sha256": sha(receipt)},
            train_rows=packet["train_rows"],
            all_actor_rows=packet["all_actor_rows"],
        )
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        result.update(status="FAILED-preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result.update(
            finished=time.time(),
            returncode=child.returncode if child else None,
            stdout_sha256=sha(records / "stdout.log")
            if (records / "stdout.log").exists()
            else None,
            stderr_sha256=sha(records / "stderr.log")
            if (records / "stderr.log").exists()
            else None,
        )
        write(records / "result.json", result)


if __name__ == "__main__":
    main()
