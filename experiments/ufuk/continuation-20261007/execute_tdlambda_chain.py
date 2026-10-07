"""ROOT typed lambda=.5 target conversion, actual8/4/fresh8 proof, then fresh64.

Each phase owns a new immutable clock; no old-clock resume or new teacher calls.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/workspace/work/harbichess/continuation-20261007")
SOURCE = BASE / "tdlambda-training-v1/source"
END = 1791448916.685839


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write(path, obj):
    with path.open("x") as stream:
        stream.write(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=[20262905, 20262906])
    parser.add_argument("--cpu-core", type=int, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    seed = args.seed
    records = BASE / f"tdlambda-actual-{seed}"
    records.mkdir(exist_ok=False)
    sources = {str(p): sha(p) for p in SOURCE.glob("*.py")}
    original = json.loads(
        Path(f"/dev/shm/harbichess-NNUE-own-contracts-v2/{seed}/fit.json").read_bytes()
    )
    data = Path(f"/dev/shm/harbichess-NNUE-tdlambda-data-v1/{seed}")
    data.mkdir(parents=True, exist_ok=False)
    first = time.time()
    receipt = dict(
        schema="ROOT-tdlambda-controlled-target-ablation-v1",
        status="running",
        first=first,
        deadline=min(first + 3000, END),
        operator_end_epoch=END,
        seed=seed,
        cpu_core=args.cpu_core,
        source_sha256=sources,
        helper_sha256=sha(__file__),
        pid=os.getpid(),
        new_teacher_queries=0,
        interpretation="new target/Adam/RNG phase; not legacy training resume",
        completed=[],
        commands=[],
    )
    write(records / "registration.json", receipt)
    sys.path.insert(0, original["core_source_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 2**30)

    def guard(deadline):
        budget.check()
        if not time.time() < min(deadline, receipt["deadline"], END):
            raise TimeoutError("same original phase/stage clock")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("disk floor256")
        if any(sha(p) != h for p, h in sources.items()):
            raise RuntimeError("frozen source changed")

    def run(command, name, deadline):
        guard(deadline)
        with (
            (records / f"{name}.stdout.log").open("xb") as out,
            (records / f"{name}.stderr.log").open("xb") as err,
        ):
            result = subprocess.run(
                command,
                stdout=out,
                stderr=err,
                timeout=max(0.001, min(deadline, receipt["deadline"]) - time.time()),
            )
        receipt["commands"].append(
            dict(
                command=command,
                returncode=result.returncode,
                finished=time.time(),
                stdout_sha256=sha(records / f"{name}.stdout.log"),
                stderr_sha256=sha(records / f"{name}.stderr.log"),
            )
        )
        if result.returncode:
            raise RuntimeError("owned child failed " + name)
        guard(deadline)

    source_paths = dict(
        source_dataset=Path(f"/dev/shm/harbichess-NNUE-own-data-v2/{seed}/dataset.json"),
        source_provenance=Path(f"/dev/shm/harbichess-NNUE-own-data-v2/{seed}/provenance.json"),
        collection_receipt=Path(f"/dev/shm/harbichess-ownq-v2/{seed}/receipt.json"),
        events=Path(f"/dev/shm/harbichess-ownq-v2/{seed}/events.jsonl"),
        conversion_result=Path(f"/dev/shm/harbichess-NNUE-own-data-v2/{seed}/result.json"),
    )
    pins = {k: sha(p) for k, p in source_paths.items()}
    provenance = json.loads(source_paths["source_provenance"].read_bytes())
    parent = original["teacher_ancestry"]["candidate"]
    parent_contract = original["teacher_ancestry"]["contract"]
    try:
        now = time.time()
        seal = dict(
            schema="own-nnue-tdlambda-target-build-seal-v1",
            status="registered",
            mode="convert",
            first=now,
            deadline=min(now + 600, END),
            operator_end_epoch=END,
            seed=seed,
            **{"lambda": 0.5},
            expected_train_rows=1024,
            inputs=pins,
            raw_collection_inputs=provenance["inputs"],
            source_files={k: ref(p) for k, p in source_paths.items()},
        )
        seal_path = records / "conversion-seal.json"
        write(seal_path, seal)
        command = [sys.executable, str(SOURCE / "prepare.py")]
        for k, p in source_paths.items():
            command += ["--" + k.replace("_", "-"), str(p)]
        command += [
            "--build-seal",
            str(seal_path),
            "--dataset-output",
            str(data / "dataset.json"),
            "--provenance-output",
            str(data / "provenance.json"),
        ]
        run(command, "convert", seal["deadline"])
        receipt["completed"].append("convert")
        for mode, cap in [("proof", 600), ("fresh-fit", 1800)]:
            now = time.time()
            phase = dict(
                schema="own-nnue-tdlambda-contract-build-seal-v1",
                status="registered",
                mode=mode,
                first=now,
                deadline=min(now + cap, END),
                operator_end_epoch=END,
                seed=seed,
                dataset=ref(data / "dataset.json"),
                provenance=ref(data / "provenance.json"),
                parent_candidate=parent,
                parent_contract=parent_contract,
                feature_schema=original["feature_schema"],
                inputs=pins,
                raw_collection_inputs=provenance["inputs"],
                prior_helper=ref(original["prior_helper_path"]),
                inference_source_sha256=original["inference_source_sha256"],
                core_source_repo=original["core_source_repo"],
                core_source_commit=original["core_source_commit"],
                collection_receipt_sha256=original["collection_receipt_sha256"],
            )
            if mode == "fresh-fit":
                phase.update(
                    tdlambda_proof_result=ref(
                        Path(f"/dev/shm/harbichess-NNUE-tdlambda-proof-v1/{seed}/result.json")
                    ),
                    tdlambda_proof_contract=ref(data / "proof-contract.json"),
                )
            phase_path = records / f"{mode}-seal.json"
            write(phase_path, phase)
            contract_path = data / (
                "proof-contract.json" if mode == "proof" else "fit-contract.json"
            )
            run(
                [
                    sys.executable,
                    str(SOURCE / "contract_builder.py"),
                    "--seal",
                    str(phase_path),
                    "--output",
                    str(contract_path),
                ],
                mode + "-contract",
                phase["deadline"],
            )
            out = Path(
                "/dev/shm/harbichess-NNUE-tdlambda-"
                + ("proof" if mode == "proof" else "fit")
                + f"-v1/{seed}"
            )
            reg = dict(
                schema="NNUE-own-tdlambda-training-orchestration-v1",
                status="registered",
                mode=mode,
                first=now,
                deadline=phase["deadline"],
                operator_end_epoch=END,
                seed=seed,
                cpu_core=args.cpu_core,
                train=ref(SOURCE / "train.py"),
                native=ref(SOURCE / "native.py"),
                contract=ref(contract_path),
                dataset=ref(data / "dataset.json"),
                target_provenance=ref(data / "provenance.json"),
                parent_candidate=parent,
                source_sha256=sources,
                inputs={
                    "seal": ref(phase_path),
                    "conversion_seal": ref(seal_path),
                    **{k: ref(p) for k, p in source_paths.items()},
                },
                output=str(out),
            )
            reg_path = records / f"{mode}-registration.json"
            write(reg_path, reg)
            run(
                [sys.executable, str(SOURCE / "prove.py"), "--registration", str(reg_path)],
                mode,
                phase["deadline"],
            )
            receipt["completed"].append(mode)
        receipt.update(
            status="PASS-actual-tdlambda-proof-and-fresh64-not-strength",
            dataset=ref(data / "dataset.json"),
            provenance=ref(data / "provenance.json"),
            proof_result=ref(
                Path(f"/dev/shm/harbichess-NNUE-tdlambda-proof-v1/{seed}/result.json")
            ),
            fit_result=ref(Path(f"/dev/shm/harbichess-NNUE-tdlambda-fit-v1/{seed}/result.json")),
            candidate=ref(
                Path(f"/dev/shm/harbichess-NNUE-tdlambda-fit-v1/{seed}/whole/candidate.pt")
            ),
        )
    except Exception as exc:
        receipt.update(status="FAILED-preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["finished"] = time.time()
        write(records / "result.json", receipt)


if __name__ == "__main__":
    main()
