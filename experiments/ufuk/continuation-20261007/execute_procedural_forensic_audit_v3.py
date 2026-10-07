"""ROOT new forensic view and six ACTUAL replays; immutable original raw receipt."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE / "human-randomstarts-own-v3-forensic-shadow"
END = 1791448916.685839


def ref(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(20262905, 20262906), required=True)
    parser.add_argument("--cpu-core", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    records = BASE / f"procedural-forensic-v3-audit-actual-{args.seed}"
    records.mkdir(exist_ok=False)
    output = Path(
        f"/dev/shm/harbichess-human-randomstarts-forensic-audit-v3/{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=False)
    registration = (
        BASE / f"human-random-collect-actual-{args.seed}/collection-registration.json"
    )
    reg = json.loads(registration.read_bytes())
    raw = Path(reg["output_path"])
    view = Path(
        f"/dev/shm/harbichess-continuation-20261007/forensic-v3-{args.seed}/receipt-view.json"
    )
    first = time.time()
    deadline = min(first + 600, END)
    commands = [
        [
            sys.executable,
            str(STAGE / "forensic_receipt.py"),
            "--registration",
            str(registration),
            "--receipt",
            str(raw / "receipt.json"),
            "--producer-directory",
            str(BASE / "human-randomstarts-own-v2"),
            "--output",
            str(view),
        ],
        [
            sys.executable,
            str(STAGE / "prepare_forensic_audit_clock.py"),
            "--root",
            str(BASE),
            "--seed",
            str(args.seed),
            "--first",
            str(first),
            "--deadline",
            str(deadline),
            "--cpu-core",
            str(args.cpu_core),
            "--helper",
            str(STAGE / "audit_collection_six_v3.py"),
            "--output",
            str(records / "clock.json"),
        ],
        [
            sys.executable,
            str(STAGE / "audit_collection_six_v3.py"),
            "--registration",
            str(registration),
            "--receipt",
            str(raw / "receipt.json"),
            "--events",
            str(raw / "events.jsonl"),
            "--forensic_view",
            str(view),
            "--clock",
            str(records / "clock.json"),
            "--output",
            str(output / "result.json"),
        ],
    ]
    receipt = dict(
        schema="ROOT-procedural-forensic-v3-actual-audit-owner-v1",
        status="running",
        first=first,
        deadline=deadline,
        operator_end_epoch=END,
        seed=args.seed,
        cpu_core=args.cpu_core,
        helper=ref(__file__),
        source_inventory=ref(STAGE / "forensic-v3-inventory.json"),
        original_registration=ref(registration),
        original_raw_receipt=ref(raw / "receipt.json"),
        commands=commands,
        completed=[],
    )
    write(records / "registration.json", receipt)
    try:
        for index, command in enumerate(commands):
            with (
                (output / f"{index}.stdout.log").open("xb") as out,
                (output / f"{index}.stderr.log").open("xb") as err,
            ):
                subprocess.run(
                    command,
                    stdout=out,
                    stderr=err,
                    check=True,
                    timeout=max(0.001, deadline - time.time()),
                )
            receipt["completed"].append(index)
        result = json.loads((output / "result.json").read_bytes())
        if (
            not result["status"].startswith("PASS-six-actual")
            or time.time() >= deadline
        ):
            raise ValueError(
                "all actual forensic six packets must PASS before original600"
            )
        if ref(raw / "receipt.json") != receipt["original_raw_receipt"]:
            raise ValueError("original raw receipt must remain unchanged")
        receipt.update(
            status="PASS-actual-six-original-parent-packets-and-forensic-binding-not-strength",
            result=ref(output / "result.json"),
            clock=ref(records / "clock.json"),
            view=ref(view),
        )
    except BaseException as error:
        receipt.update(status="FAILED-preserved", error=repr(error))
        raise
    finally:
        receipt["finished"] = time.time()
        write(records / "result.json", receipt)


if __name__ == "__main__":
    main()
