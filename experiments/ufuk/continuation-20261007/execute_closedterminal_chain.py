"""ROOT-clocked closed-terminal conversion, six actual search audits, proof and fresh64."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE / "closed-terminal-own-v1"
SOURCE = ROOT / "source"
END = 1791448916.685839


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=[20262905, 20262906], required=True)
    parser.add_argument("--cpu-core", type=int, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    spec = importlib.util.spec_from_file_location(
        "root_MC_factory", ROOT / "integration/seal_factory.py"
    )
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    seed = args.seed
    records = BASE / f"closedterminal-actual-{seed}"
    records.mkdir(exist_ok=False)
    original = json.loads(
        Path(f"/dev/shm/harbichess-NNUE-own-contracts-v2/{seed}/fit.json").read_bytes()
    )
    parent = original["teacher_ancestry"]
    source_pins = {str(p): factory.sha(p) for p in SOURCE.glob("*.py")}
    source_pins[str(ROOT / "integration/audit_six.py")] = factory.sha(
        ROOT / "integration/audit_six.py"
    )
    source_pins[str(ROOT / "integration/seal_factory.py")] = factory.sha(
        ROOT / "integration/seal_factory.py"
    )
    first = time.time()
    record = dict(
        schema="ROOT-closedterminal-conversion-audit-proof-fresh64-v1",
        status="running",
        first=first,
        deadline=min(first + 3000, END),
        seed=seed,
        cpu_core=args.cpu_core,
        source_pins=source_pins,
        helper=factory.ref(__file__),
        new_teacher_queries=0,
        commands=[],
    )
    factory.write_once(records / "registration.json", record)
    sys.path.insert(0, original["core_source_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard(deadline):
        memory.check()
        if not time.time() < min(deadline, record["deadline"], END):
            raise TimeoutError("original immutable stage/phase clocks")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("disk floor256MiB")
        if any(factory.sha(p) != h for p, h in source_pins.items()):
            raise RuntimeError("frozen source changed")

    def run(command, name, deadline):
        guard(deadline)
        with (
            (records / (name + ".stdout.log")).open("xb") as out,
            (records / (name + ".stderr.log")).open("xb") as err,
        ):
            r = subprocess.run(
                command,
                stdout=out,
                stderr=err,
                timeout=min(deadline, record["deadline"]) - time.time(),
            )
        record["commands"].append(
            dict(
                command=command,
                returncode=r.returncode,
                finished=time.time(),
                stdout=factory.ref(records / (name + ".stdout.log")),
                stderr=factory.ref(records / (name + ".stderr.log")),
            )
        )
        if r.returncode:
            raise RuntimeError("owned child failed " + name)
        guard(deadline)

    data = Path(f"/dev/shm/harbichess-NNUE-closedterminal-data-v1/{seed}")
    try:
        now = time.time()
        conversion = factory.make_conversion_spec(
            BASE / f"closedterminal-collection-{seed}/registration.json",
            Path(f"/dev/shm/harbichess-closedterminal-v1/{seed}/receipt.json"),
            first=now,
            deadline=min(now + 600, END),
            core_repo=original["core_source_repo"],
        )
        conversion_path = records / "conversion-seal.json"
        factory.write_once(conversion_path, conversion)
        run(
            [
                sys.executable,
                str(SOURCE / "convert.py"),
                "--seal",
                str(conversion_path),
                "--output",
                str(data),
            ],
            "conversion",
            conversion["deadline"],
        )
        now = time.time()
        audit_helper = ROOT / "integration/audit_six.py"
        audit_clock = dict(
            schema="NNUE-own-closed-terminal-six-search-audit-clock-v1",
            status="registered",
            first=now,
            deadline=min(now + 600, END),
            cpu_core=args.cpu_core,
            helper_sha256=factory.sha(audit_helper),
            spec_sha256=factory.sha(conversion_path),
            registration_sha256=conversion["registration"]["sha256"],
            receipt_sha256=conversion["receipt"]["sha256"],
            events_sha256=conversion["events"]["sha256"],
            converter_sha256=factory.sha(SOURCE / "convert.py"),
        )
        audit_clock_path = records / "six-audit-clock.json"
        factory.write_once(audit_clock_path, audit_clock)
        run(
            [
                sys.executable,
                str(audit_helper),
                "--spec",
                str(conversion_path),
                "--clock",
                str(audit_clock_path),
                "--output",
                str(data / "six-actual-search-audit.json"),
            ],
            "six-actual-search-audit",
            audit_clock["deadline"],
        )
        for mode, duration in [("proof", 600), ("fresh-fit", 1800)]:
            now = time.time()
            proof_out = Path(f"/dev/shm/harbichess-NNUE-closedterminal-proof-v1/{seed}")
            fit_out = Path(f"/dev/shm/harbichess-NNUE-closedterminal-fit-v1/{seed}")
            kwargs = dict(
                mode=mode,
                first=now,
                deadline=min(now + duration, END),
                operator_end_epoch=END,
                seed=seed,
                conversion_spec=conversion_path,
                conversion_result=data / "result.json",
                parent_contract=parent["contract"]["path"],
                core_source_repo=original["core_source_repo"],
                core_source_commit=original["core_source_commit"],
                feature_schema=original["feature_schema"],
            )
            if mode == "fresh-fit":
                kwargs.update(
                    proof_result=proof_out / "result.json",
                    proof_contract=data / "proof-contract.json",
                )
            seal = factory.make_seal(**kwargs)
            seal_path = records / (mode + "-seal.json")
            factory.write_once(seal_path, seal)
            contract_path = data / (
                "proof-contract.json" if mode == "proof" else "fit-contract.json"
            )
            run(
                [
                    sys.executable,
                    str(SOURCE / "contract_builder.py"),
                    "--seal",
                    str(seal_path),
                    "--output",
                    str(contract_path),
                ],
                mode + "-contract",
                seal["deadline"],
            )
            c = json.loads(contract_path.read_bytes())
            registration = dict(
                schema="NNUE-own-closed-terminal-training-orchestration-v1",
                status="registered",
                mode=mode,
                first=now,
                deadline=seal["deadline"],
                operator_end_epoch=END,
                seed=seed,
                cpu_core=args.cpu_core,
                train=factory.ref(SOURCE / "train.py"),
                native=factory.ref(SOURCE / "native.py"),
                contract=factory.ref(contract_path),
                dataset=factory.ref(data / "dataset.json"),
                target_provenance=factory.ref(data / "provenance.json"),
                parent_candidate=parent["candidate"],
                source_sha256=c["source_sha256"] | c["execution_helpers_sha256"],
                inputs={
                    "build_seal": factory.ref(seal_path),
                    "conversion_seal": factory.ref(conversion_path),
                    "contract": factory.ref(contract_path),
                    "dataset": factory.ref(data / "dataset.json"),
                    "target_provenance": factory.ref(data / "provenance.json"),
                    "parent_candidate": parent["candidate"],
                },
                output=str(proof_out if mode == "proof" else fit_out),
            )
            registration_path = records / (mode + "-registration.json")
            factory.write_once(registration_path, registration)
            run(
                [
                    sys.executable,
                    str(SOURCE / "prove.py"),
                    "--registration",
                    str(registration_path),
                ],
                mode,
                seal["deadline"],
            )
        record.update(
            status="PASS-actual-closedterminal-audit-proof-and-fresh64-not-strength",
            dataset=factory.ref(data / "dataset.json"),
            provenance=factory.ref(data / "provenance.json"),
            proof_result=factory.ref(proof_out / "result.json"),
            fit_result=factory.ref(fit_out / "result.json"),
            candidate=factory.ref(fit_out / "whole/candidate.pt"),
        )
    except BaseException as error:
        record.update(status="FAILED-preserved", error=repr(error))
        raise
    finally:
        record["finished"] = time.time()
        factory.write_once(records / "result.json", record)


if __name__ == "__main__":
    main()
