"""ROOT full replay conversion after both actual forensic six-search audits."""

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
AUDIT_SET = BASE / "procedural-forensic-v4-actual-audit-set.json"


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(20262905, 20262906), required=True)
    parser.add_argument("--cpu-core", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    for seed in (20262905, 20262906):
        result = json.loads(
            (
                BASE / f"procedural-forensic-v3-audit-actual-{seed}/result.json"
            ).read_bytes()
        )
        if (
            result["status"]
            != "PASS-actual-six-original-parent-packets-and-forensic-binding-not-strength"
        ):
            raise ValueError("both actual forensic audits required before conversion")
    records = BASE / f"procedural-forensic-v4-conversion-actual-{args.seed}"
    records.mkdir(exist_ok=False)
    output = Path(
        f"/dev/shm/harbichess-human-randomstarts-forensic-data-v4/{args.seed}"
    )
    logs = Path(
        f"/dev/shm/harbichess-continuation-20261007/procedural-forensic-v4-conversion-logs-{args.seed}"
    )
    logs.mkdir(parents=True, exist_ok=False)
    first = time.time()
    deadline = min(first + 600, END)
    commands = [
        [
            sys.executable,
            str(STAGE / "prepare_forensic_conversion_v3.py"),
            "--root",
            str(BASE),
            "--seed",
            str(args.seed),
            "--audit-set",
            str(AUDIT_SET),
            "--first",
            str(first),
            "--deadline",
            str(deadline),
            "--validator",
            str(STAGE / "forensic_audit_set.py"),
            "--runtime",
            str(STAGE / "audit_collection_six_v3.py"),
            "--output",
            str(records / "conversion-seal.json"),
        ],
        [
            sys.executable,
            str(STAGE / "convert_forensic_v3.py"),
            "--seal",
            str(records / "conversion-seal.json"),
            "--output",
            str(output),
        ],
    ]
    receipt = dict(
        schema="ROOT-procedural-forensic-v3-full-conversion-owner-v1",
        status="running",
        first=first,
        deadline=deadline,
        operator_end_epoch=END,
        seed=args.seed,
        cpu_core=args.cpu_core,
        helper=ref(__file__),
        source_inventory=ref(STAGE / "forensic-v3-inventory.json"),
        audit_set=ref(AUDIT_SET),
        commands=commands,
        completed=[],
    )
    write(records / "registration.json", receipt)
    try:
        for i, command in enumerate(commands):
            with (
                (logs / f"{i}.stdout.log").open("xb") as out,
                (logs / f"{i}.stderr.log").open("xb") as err,
            ):
                subprocess.run(
                    command,
                    stdout=out,
                    stderr=err,
                    check=True,
                    timeout=max(0.001, deadline - time.time()),
                )
            receipt["completed"].append(i)
        result = json.loads((output / "result.json").read_bytes())
        if (
            result["status"] != "PASS-own1024-fullhistory-trace-conversion-not-strength"
            or time.time() >= deadline
        ):
            raise ValueError("complete original-clock replay conversion required")
        receipt.update(
            status="PASS-full1024-forensic-own-conversion-not-strength",
            conversion_result=ref(output / "result.json"),
            seal=ref(records / "conversion-seal.json"),
        )
    except BaseException as error:
        receipt.update(status="FAILED-preserved", error=repr(error))
        raise
    finally:
        receipt["finished"] = time.time()
        write(records / "result.json", receipt)


if __name__ == "__main__":
    main()
