"""ROOT closed procedural own-data audit, conversion, proof, and fresh-fit phases."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE / "human-randomstarts-own-v2"
END = 1791448916.685839
SEEDS = (20262905, 20262906)


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def actual(seed, phase):
    path = BASE / f"procedural-own-{phase}-actual-{seed}" / "result.json"
    data = json.loads(path.read_bytes())
    if data["status"] != "PASS-actual-" + phase + "-not-strength":
        raise ValueError("closed actual phase must PASS: " + str(path))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument(
        "--phase", choices=("audit", "convert", "proof", "fit"), required=True
    )
    parser.add_argument("--cpu-core", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    reg_path = (
        BASE
        / f"human-random-collect-actual-{args.seed}"
        / "collection-registration.json"
    )
    reg = json.loads(reg_path.read_bytes())
    closed = json.loads((reg_path.parent / "result.json").read_bytes())
    if closed["status"] != "PASS-actual-procedural-own-Q-collection-not-strength":
        raise ValueError("original actor must close normally before new phase")
    raw = Path(reg["output_path"])
    receipt = json.loads((raw / "receipt.json").read_bytes())
    if receipt["status"] != "PASS-exact-row-budget" or receipt["train_rows"] != 1024:
        raise ValueError("original chronological1024 budget")
    records = BASE / f"procedural-own-{args.phase}-actual-{args.seed}"
    records.mkdir(exist_ok=False)
    logs = Path("/dev/shm/harbichess-continuation-20261007") / records.name
    logs.mkdir(exist_ok=False)
    first = time.time()
    deadline = min(first + (1800 if args.phase == "fit" else 600), END)
    result = dict(
        schema="ROOT-procedural-own-learning-phase-v2",
        status="running",
        seed=args.seed,
        phase=args.phase,
        first=first,
        deadline=deadline,
        operator_end_epoch=END,
        cpu_core=args.cpu_core,
        helper=ref(__file__),
        collection_registration=ref(reg_path),
        collection_receipt=ref(raw / "receipt.json"),
        source_inventory=ref(STAGE / "source-inventory.json"),
        commands=[],
        teacher_labels_used=False,
        trained_teacher_weights_used=False,
    )
    write(records / "registration.json", result)

    def run(name, command):
        out, err = logs / (name + ".stdout.log"), logs / (name + ".stderr.log")
        with out.open("xb") as stdout, err.open("xb") as stderr:
            process = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                timeout=max(0.001, deadline - time.time()),
            )
        result["commands"].append(
            dict(
                name=name,
                command=command,
                returncode=process.returncode,
                finished=time.time(),
                stdout=ref(out),
                stderr=ref(err),
            )
        )
        if process.returncode:
            raise RuntimeError("owned phase failed: " + name)
        if time.time() >= deadline:
            raise TimeoutError("original immutable phase deadline")

    try:
        if args.phase == "audit":
            clock = dict(
                schema="human-randomstarts-own-six-root-audit-clock-v2",
                helper_sha256=ref(STAGE / "audit_collection_six.py")["sha256"],
                registration_sha256=ref(reg_path)["sha256"],
                receipt_sha256=ref(raw / "receipt.json")["sha256"],
                events_sha256=ref(raw / "events.jsonl")["sha256"],
                cpu_core=args.cpu_core,
                producer_directory=str(STAGE),
                first=first,
                deadline=deadline,
            )
            write(records / "clock.json", clock)
            run(
                "six-actual-replays",
                [
                    sys.executable,
                    str(STAGE / "audit_collection_six.py"),
                    "--registration",
                    str(reg_path),
                    "--receipt",
                    str(raw / "receipt.json"),
                    "--events",
                    str(raw / "events.jsonl"),
                    "--clock",
                    str(records / "clock.json"),
                    "--output",
                    str(records / "six-actual-replays.json"),
                ],
            )
            result["six_replays"] = ref(records / "six-actual-replays.json")
            result["clock"] = ref(records / "clock.json")
        else:
            actual(args.seed, "audit")
            spec = records / "spec"
            command = [
                sys.executable,
                str(STAGE / "specs.py"),
                "--phase",
                args.phase,
                "--seed",
                str(args.seed),
                "--registration",
                str(reg_path),
                "--training",
                str(STAGE),
                "--first",
                str(first),
                "--deadline",
                str(deadline),
                "--operator-end",
                str(END),
                "--output",
                str(spec),
            ]
            if args.phase == "convert":
                command += ["--producer", str(STAGE)]
            else:
                actual(args.seed, "convert")
                audits_path = BASE / "procedural-own-actual-audit-set.json"
                registrations_path = BASE / "procedural-own-actual-registrations.json"
                if not audits_path.is_file() or not registrations_path.is_file():
                    raise ValueError(
                        "publish BOTH actual six-search audits and registration PATH STRINGS first"
                    )
                for seed in SEEDS:
                    actual(seed, "audit")
                command += [
                    "--registrations",
                    str(registrations_path),
                    "--audits",
                    str(audits_path),
                    "--cpu-core",
                    str(args.cpu_core),
                ]
                if args.phase == "fit":
                    for seed in SEEDS:
                        actual(seed, "proof")
                    proof = actual(args.seed, "proof")
                    command += [
                        "--own-proof-result",
                        proof["native_result"]["path"],
                        "--own-proof-contract",
                        proof["contract"]["path"],
                    ]
            run("seal-spec", command)
            for index, command in enumerate(
                json.loads((spec / "commands.json").read_bytes())
            ):
                run(f"sealed-command-{index}", command)
            if args.phase == "convert":
                data = (
                    Path("/dev/shm/harbichess-human-randomstarts-own-data-v2")
                    / f"g-{reg['generation']}"
                    / str(args.seed)
                )
                result.update(
                    dataset=ref(data / "dataset.json"),
                    provenance=ref(data / "provenance.json"),
                )
            else:
                registry = spec.with_name(spec.name + "-registry")
                registered = json.loads((registry / "registration.json").read_bytes())
                for index, command in enumerate(
                    json.loads((registry / "commands.json").read_bytes())
                ):
                    expected = [
                        sys.executable,
                        str(STAGE / "prove.py"),
                        "--registration",
                        str(registry / "registration.json"),
                    ]
                    if command != expected:
                        raise ValueError(
                            "exact original procedural native command required"
                        )
                    adapter = (
                        BASE / "procedural-proof-registration-adapter-v1" / "adapter.py"
                    )
                    adapted = [
                        sys.executable,
                        str(adapter),
                        "--registration",
                        str(registry / "registration.json"),
                        "--registration-sha256",
                        ref(registry / "registration.json")["sha256"],
                        "--proof-helper",
                        str(STAGE / "prove.py"),
                        "--proof-helper-sha256",
                        ref(STAGE / "prove.py")["sha256"],
                        "--derivation-receipt",
                        str(logs / "native-projection-receipt.json"),
                    ]
                    run(f"native-command-{index}-explicit-typed-projection", adapted)
                    result["integration_adapter"] = ref(adapter)
                    result["integration_projection"] = ref(
                        logs / "native-projection-receipt.json"
                    )
                result.update(
                    native_registration=ref(registry / "registration.json"),
                    contract=registered["contract"],
                    native_result=ref(Path(registered["output"]) / "result.json"),
                )
        result["status"] = "PASS-actual-" + args.phase + "-not-strength"
    except BaseException as error:
        result.update(status="FAILED-preserved", error=repr(error))
        raise
    finally:
        result["finished"] = time.time()
        write(records / "result.json", result)


if __name__ == "__main__":
    main()
