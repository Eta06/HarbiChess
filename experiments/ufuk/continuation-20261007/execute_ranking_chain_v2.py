"""ROOT-clocked selected-action ranking conversion, full-native proof and fixed fresh64."""

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
ROOT = BASE / "action-ranking-own-v2-source-binding"
SOURCE = ROOT / "source"
END = 1791448916.685839


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=[20262905, 20262906], required=True)
    parser.add_argument("--cpu-core", type=int, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    spec = importlib.util.spec_from_file_location(
        "root_ranking_factory", ROOT / "integration/seal_factory.py"
    )
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    seed = args.seed
    records = BASE / f"ranking-recovery-actual-{seed}"
    records.mkdir(exist_ok=False)
    original = json.loads(
        Path(f"/dev/shm/harbichess-NNUE-own-contracts-v2/{seed}/fit.json").read_bytes()
    )
    parent = original["teacher_ancestry"]
    source_pins = {str(p): factory.sha(p) for p in SOURCE.glob("*.py")}
    source_pins[str(ROOT / "integration/seal_factory.py")] = factory.sha(
        ROOT / "integration/seal_factory.py"
    )
    legacy = BASE / f"ranking-actual-{seed}/registration.json"
    first = json.loads(legacy.read_bytes())["first"] if legacy.exists() else time.time()
    record = dict(
        schema="ROOT-ranking-conversion-proof-fresh64-v1",
        status="running",
        first=first,
        deadline=min(first + 3000, END),
        seed=seed,
        cpu_core=args.cpu_core,
        source_pins=source_pins,
        helper=factory.ref(__file__),
        new_teacher_queries=0,
        commands=[],
        recovery_observed_epoch=time.time(),
        failed_attempt=str(legacy) if legacy.exists() else None,
        old_proof_clock_preserved=True,
        existing_converted_data_unchanged=True,
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

    data = Path(f"/dev/shm/harbichess-NNUE-ranking-data-v1/{seed}")
    try:
        if not data.exists():
            now = time.time()
            conversion = factory.make_conversion_seal(
                afterstate_seal=BASE / f"afterstate-actual-{seed}/conversion-seal.json",
                afterstate_result=Path(
                    f"/dev/shm/harbichess-NNUE-afterstate-data-v1/{seed}/result.json"
                ),
                afterstate_converter=BASE / "afterstate-own-v1/source/convert.py",
                first=now,
                deadline=min(now + 600, END),
                operator_end_epoch=END,
                seed=seed,
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
        for mode, duration in [("proof", 600), ("fresh-fit", 1800)]:
            old_proof = BASE / f"ranking-actual-{seed}/proof-registration.json"
            now = (
                json.loads(old_proof.read_bytes())["first"]
                if mode == "proof" and old_proof.exists()
                else time.time()
            )
            proof_out = Path(
                f"/dev/shm/harbichess-NNUE-ranking-recovery-proof-v2/{seed}"
            )
            fit_out = Path(f"/dev/shm/harbichess-NNUE-ranking-recovery-fit-v2/{seed}")
            kwargs = dict(
                mode=mode,
                first=now,
                deadline=min(now + duration, END),
                operator_end_epoch=END,
                seed=seed,
                dataset=data / "dataset.json",
                provenance=data / "provenance.json",
                parent_candidate=parent["candidate"]["path"],
                parent_contract=parent["contract"]["path"],
                feature_schema=original["feature_schema"],
                prior_helper=original["prior_helper_path"],
                inference_source_sha256=original["inference_source_sha256"],
                core_source_repo=original["core_source_repo"],
                core_source_commit=original["core_source_commit"],
            )
            if mode == "fresh-fit":
                kwargs.update(
                    proof_result=proof_out / "result.json",
                    proof_contract=data / "proof-contract.json",
                )
            seal = factory.make_build_seal(**kwargs)
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
            registration = factory.make_orchestration(
                mode=mode,
                first=now,
                deadline=seal["deadline"],
                operator_end_epoch=END,
                seed=seed,
                cpu_core=args.cpu_core,
                contract=contract_path,
                dataset=data / "dataset.json",
                provenance=data / "provenance.json",
                parent_candidate=parent["candidate"]["path"],
                build_seal=seal_path,
                source_dir=SOURCE,
                output=proof_out if mode == "proof" else fit_out,
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
            status="PASS-actual-ranking-proof-and-fresh64-not-strength",
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
