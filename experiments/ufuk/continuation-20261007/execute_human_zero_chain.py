"""ROOT actual teacher-free zero initialization, admission and synthetic native proof."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE / "human-prior-own-v1"
CORE = Path("/workspace/work/harbichess/cpu-additive-source-6fcc8b4")
INFERENCE = Path("/workspace/work/harbichess/cpu-kingbucket-nnue-proposal")
END = 1791448916.685839


def ref(p):
    return dict(path=str(p), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())


def write(p, j):
    with p.open("x") as f:
        json.dump(j, f, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=[20262905, 20262906], required=True)
    parser.add_argument("--cpu-core", type=int, choices=[1, 3], required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    records = BASE / f"human-zero-actual-{args.seed}"
    records.mkdir(exist_ok=False)
    zero = Path(f"/dev/shm/harbichess-human-prior-zero-v1/{args.seed}")
    first = time.time()
    record = dict(
        schema="ROOT-actual-human-zero-and-synthetic-native-proof-v1",
        status="running",
        first=first,
        deadline=min(first + 1800, END),
        seed=args.seed,
        cpu_core=args.cpu_core,
        helper=ref(__file__),
        source_inventory=ref(STAGE / "source-inventory.json"),
        trained_teacher_weights_used=False,
        teacher_label_calls=0,
        optimizer_updates_on_actual_data=0,
        commands=[],
    )
    write(records / "registration.json", record)

    def run(name, command, deadline):
        with (
            (records / (name + ".stdout.log")).open("xb") as out,
            (records / (name + ".stderr.log")).open("xb") as err,
        ):
            result = subprocess.run(
                command,
                stdout=out,
                stderr=err,
                timeout=min(deadline, record["deadline"]) - time.time(),
            )
        record["commands"].append(
            dict(
                command=command,
                returncode=result.returncode,
                finished=time.time(),
                stdout=ref(records / (name + ".stdout.log")),
                stderr=ref(records / (name + ".stderr.log")),
            )
        )
        if result.returncode:
            raise RuntimeError("owned phase failed " + name)

    try:
        now = time.time()
        seal = records / "initialization-seal.json"
        prior = Path(
            "/workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/value.py"
        )
        command = [
            sys.executable,
            str(STAGE / "prepare_zero.py"),
            "--seed",
            str(args.seed),
            "--first",
            str(now),
            "--deadline",
            str(now + 600),
            "--operator-end-epoch",
            str(END),
            "--core-repo",
            str(CORE),
            "--core-commit",
            "6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
            "--prior-helper",
            str(prior),
            "--search-helper",
            "/workspace/HarbiChess/experiments/ufuk/cpu-budget-search-v1/search.py",
            "--output",
            str(seal),
        ]
        for p in [
            INFERENCE / n
            for n in [
                "model.py",
                "native.py",
                "evaluator.py",
                "forward.c",
                "_kingbucket16.cpython-312-x86_64-linux-gnu.so",
                "build-receipt.json",
            ]
        ] + [prior]:
            command += ["--inference-file", str(p)]
        run("prepare-zero", command, now + 600)
        run(
            "initialize",
            [
                sys.executable,
                str(STAGE / "initialize.py"),
                "--seal",
                str(seal),
                "--output",
                str(zero),
            ],
            now + 600,
        )
        admission_seal = records / "parent-admission-seal.json"
        now = time.time()
        run(
            "parent-seal",
            [
                sys.executable,
                str(STAGE / "zero_parent.py"),
                "--initialization-result",
                str(zero / "result.json"),
                "--output",
                str(admission_seal),
            ],
            now + 600,
        )
        admission_clock = records / "parent-admission-clock.json"
        write(
            admission_clock,
            dict(
                schema="human-prior-own-parent-readonly-admission-clock-v1",
                status="registered",
                first=now,
                deadline=now + 600,
                operator_end_epoch=END,
                seal_sha256=ref(admission_seal)["sha256"],
                helper_sha256=ref(STAGE / "admit_parent.py")["sha256"],
            ),
        )
        admission_result = records / "parent-admission-result.json"
        run(
            "parent-admission",
            [
                sys.executable,
                str(STAGE / "admit_parent.py"),
                "--seal",
                str(admission_seal),
                "--clock",
                str(admission_clock),
                "--output",
                str(admission_result),
            ],
            now + 600,
        )
        now = time.time()
        qualification = records / "synthetic-qualification-registration.json"
        command = [
            sys.executable,
            str(STAGE / "qualification/qualify.py"),
            "--prepare-observed",
            "--initialization-result",
            str(zero / "result.json"),
            "--seed",
            str(args.seed),
            "--cpu-core",
            str(args.cpu_core),
            "--first",
            str(now),
            "--deadline",
            str(now + 600),
            "--operator-end",
            str(END),
            "--stage",
            str(STAGE),
            "--inventory",
            str(STAGE / "source-inventory.json"),
            "--proof-output",
            f"/dev/shm/harbichess-human-prior-synthetic-v1/{args.seed}",
            "--output",
            str(qualification),
        ]
        run("prepare-synthetic-qualification", command, now + 600)
        run(
            "synthetic-native-qualification",
            [
                sys.executable,
                str(STAGE / "qualification/qualify.py"),
                "--registration",
                str(qualification),
            ],
            now + 600,
        )
        record.update(
            status="PASS-actual-zero-init-admission-and-synthetic-native-proof-not-strength",
            zero_result=ref(zero / "result.json"),
            parent_admission_seal=ref(admission_seal),
            parent_admission_result=ref(admission_result),
        )
    except BaseException as error:
        record.update(status="FAILED-preserved", error=repr(error))
        raise
    finally:
        record["finished"] = time.time()
        write(records / "result.json", record)


if __name__ == "__main__":
    main()
