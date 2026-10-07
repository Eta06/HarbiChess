"""ROOT search-only human-prior profile/known32, no training or promotion."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE / "progressive-width-search-v1"
END = 1791448916.685839
CORE = Path("/workspace/work/harbichess/cpu-additive-source-6fcc8b4")


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("profile", "arena"), required=True)
    parser.add_argument("--cpu-core", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    records = BASE / f"progressive-width-actual-{args.mode}-v1"
    records.mkdir(exist_ok=False)
    output = Path(f"/dev/shm/harbichess-progressive-width-{args.mode}-v1")
    logs = Path(
        f"/dev/shm/harbichess-continuation-20261007/progressive-width-{args.mode}-logs-v1"
    )
    logs.mkdir(parents=True, exist_ok=False)
    first = time.time()
    deadline = min(first + (600 if args.mode == "profile" else 7200), END)
    registration = records / "development-registration.json"
    command = [
        sys.executable,
        str(STAGE / "qualification/register.py"),
        "--mode",
        args.mode,
        "--first",
        str(first),
        "--deadline",
        str(deadline),
        "--operator-end-epoch",
        str(END),
        "--cpu-core",
        str(args.cpu_core),
        "--core-repo",
        str(CORE),
        "--core-commit",
        "6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
        "--old-search",
        str(BASE / "tdleaf-search-equivalence-v3/search_original.py"),
        "--advanced-search",
        str(STAGE / "search.py"),
        "--prior-helper",
        "/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/value.py",
        "--output",
        str(output),
        "--registration",
        str(registration),
    ]
    if args.mode == "profile":
        command += [
            "--train-roots",
            str(BASE / "nnue-own-producer-v2/input-pools/20262905.json"),
        ]
    else:
        command += [
            "--profile-result",
            "/dev/shm/harbichess-progressive-width-profile-v1/result.json",
            "--book",
            "/workspace/work/harbichess/cpu-own-replay-v1-actual/development/book.json",
            "--stockfish",
            "/workspace/work/harbichess/stockfish/stockfish-linux-x86-64-universal",
        ]
    commands = [
        command,
        [
            sys.executable,
            str(STAGE / "qualification/develop.py"),
            "--registration",
            str(registration),
        ],
    ]
    receipt = dict(
        schema="ROOT-progressive-width-search-only-owner-v1",
        status="running",
        mode=args.mode,
        first=first,
        deadline=deadline,
        operator_end_epoch=END,
        source_readiness=ref(STAGE / "readiness.json"),
        helper=ref(__file__),
        cpu_core=args.cpu_core,
        commands=commands,
        completed=[],
        selflearning_claim=False,
    )
    write(records / "registration.json", receipt)
    try:
        for i, cmd in enumerate(commands):
            with (
                (logs / f"{i}.stdout.log").open("xb") as out,
                (logs / f"{i}.stderr.log").open("xb") as err,
            ):
                subprocess.run(
                    cmd,
                    stdout=out,
                    stderr=err,
                    check=True,
                    timeout=max(0.001, deadline - time.time()),
                )
            receipt["completed"].append(i)
        result = json.loads((output / "result.json").read_bytes())
        expected = (
            "PASS-24-paired-TRAIN-searches-not-strength"
            if args.mode == "profile"
            else "PASS-32-fixed-development-games-search-only-not-selflearning"
        )
        if result["status"] != expected or time.time() >= deadline:
            raise ValueError("actual original-clock profile/development required")
        if args.mode == "profile" and not result["latency_within1p10"]:
            raise ValueError("unchanged1.10 speed admission")
        receipt.update(
            status="PASS-actual-search-only-development-not-selflearning",
            result=ref(output / "result.json"),
        )
    except BaseException as error:
        receipt.update(status="FAILED-preserved", error=repr(error))
        raise
    finally:
        receipt["finished"] = time.time()
        write(records / "result.json", receipt)


if __name__ == "__main__":
    main()
