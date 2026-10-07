"""ROOT-run one forensic v4 native proof or fresh64 phase after replay conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from forensic_receipt import canonical

END = 1791448916.685839
SEEDS = (20262905, 20262906)
HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def phase_paths(seed: int, mode: str) -> tuple[Path, Path]:
    base = Path("/dev/shm/harbichess-human-randomstarts-forensic-native-v4") / str(seed)
    return base / mode, base / f"{mode}-contract.json"


def execute(root: Path, seed: int, mode: str, cpu_core: int) -> dict:
    root = Path(root).resolve(strict=True)
    if seed not in SEEDS or mode not in ("proof", "fresh-fit") or cpu_core not in (1, 3):
        raise ValueError("fixed seed/phase/core")
    os.sched_setaffinity(0, {cpu_core})
    phase_first = time.time()
    duration = 600 if mode == "proof" else 1800
    deadline = min(phase_first + duration, END)
    if not phase_first < deadline:
        raise TimeoutError("operator end reached")

    phase_dir = root / f"forensic-own-v4-actual-{seed}" / mode
    phase_dir.mkdir(parents=True, exist_ok=False)
    data_dir = Path(f"/dev/shm/harbichess-human-randomstarts-forensic-data-v4/{seed}")
    dataset = data_dir / "dataset.json"
    provenance = data_dir / "provenance.json"
    conversion_result = data_dir / "result.json"
    audit_set = root / "procedural-forensic-v4-actual-audit-set.json"
    validator = Path(
        "/workspace/work/harbichess/continuation-20261007/"
        "human-randomstarts-own-v3-forensic-shadow/forensic_audit_set.py"
    )
    runtime = Path(
        "/workspace/work/harbichess/continuation-20261007/"
        "human-randomstarts-own-v3-forensic-shadow/audit_collection_six_v3.py"
    )
    parent = root / f"human-random-collect-actual-{seed}/collection-registration.json"
    for path in (dataset, provenance, conversion_result, audit_set, validator, runtime, parent):
        if not path.is_file():
            raise FileNotFoundError(path)
    converted = json.loads(conversion_result.read_bytes())
    if (
        converted.get("status") != "PASS-own1024-fullhistory-trace-conversion-not-strength"
        or converted.get("dataset_sha256") != sha(dataset)
        or converted.get("provenance_sha256") != sha(provenance)
    ):
        raise ValueError("actual full replay conversion result must bind both output files")

    native_base = Path("/dev/shm/harbichess-human-randomstarts-forensic-native-v4")
    proof_result = native_base / str(seed) / "proof/result.json"
    proof_contract = Path(
        native_base / str(seed) / "proof-contract.json"
    )
    expected_status = "PASS-own-NNUE-fixed-phase-and-fresh-native-loads-not-strength"
    if mode == "fresh-fit":
        proof_owner_status = "PASS-forensic-v4-whole-pause-fresh-native-proof-not-strength"
        for proof_seed in SEEDS:
            owner_path = root / f"forensic-own-v4-actual-{proof_seed}/proof/result.json"
            native_path = native_base / str(proof_seed) / "proof/result.json"
            owner_result = json.loads(owner_path.read_bytes())
            prior_result = json.loads(native_path.read_bytes())
            if owner_result.get("status") != proof_owner_status:
                raise ValueError(f"both actual proof phases must pass before fresh64: {proof_seed}")
            if (
                prior_result.get("status") != expected_status
                or prior_result.get("full_payload_bits_equal") is not True
            ):
                raise ValueError(f"whole/pause/resume proof bits required for {proof_seed}")

    output, contract_path = phase_paths(seed, mode)
    log_dir = Path(
        f"/dev/shm/harbichess-continuation-20261007/forensic-native-v4-logs/{seed}/{mode}"
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    operator_end = json.loads(parent.read_bytes())["operator_end_epoch"]
    record = {
        "schema": "ROOT-forensic-v4-native-proof-fit-phase-v1",
        "status": "running",
        "seed": seed,
        "mode": mode,
        "first": phase_first,
        "deadline": deadline,
        "operator_end_epoch": operator_end,
        "cpu_core": cpu_core,
        "helper": ref(Path(__file__)),
        "source_dir": str(HERE),
        "conversion": ref(conversion_result),
        "dataset": ref(dataset),
        "provenance": ref(provenance),
        "audit_set": ref(audit_set),
        "validator": ref(validator),
        "audit_runtime": ref(runtime),
        "commands": [],
    }
    write_once(phase_dir / "registration.json", record)

    sys.path.insert(0, str(root / "../cpu-additive-source-6fcc8b4/src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard() -> None:
        memory.check()
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("disk floor 256 MiB")
        if not phase_first <= time.time() < deadline <= END:
            raise TimeoutError("original phase clock elapsed")

    def run(name: str, command: list[str]) -> None:
        guard()
        out_path, err_path = log_dir / f"{name}.stdout.log", log_dir / f"{name}.stderr.log"
        with out_path.open("xb") as out, err_path.open("xb") as err:
            result = subprocess.run(
                command,
                cwd=HERE,
                stdout=out,
                stderr=err,
                check=False,
                timeout=max(0.001, deadline - time.time()),
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        command_record = {
            "name": name,
            "command": command,
            "returncode": result.returncode,
            "finished": time.time(),
            "stdout": ref(out_path),
            "stderr": ref(err_path),
        }
        record["commands"].append(command_record)
        if result.returncode:
            raise RuntimeError(f"forensic v4 phase child failed: {name}")
        guard()

    try:
        seal_path = phase_dir / "contract-build-seal.json"
        registration_path = phase_dir / "training-registration.json"
        command = [
            sys.executable,
            str(HERE / "prepare_forensic_contract_v4.py"),
            "--root",
            str(root),
            "--seed",
            str(seed),
            "--mode",
            mode,
            "--first",
            str(phase_first),
            "--deadline",
            str(deadline),
            "--dataset",
            str(dataset),
            "--provenance",
            str(provenance),
            "--audit-set",
            str(audit_set),
            "--validator",
            str(validator),
            "--output",
            str(seal_path),
        ]
        if mode == "fresh-fit":
            command += [
                "--proof-result",
                str(proof_result),
                "--proof-contract",
                str(proof_contract),
            ]
        run("prepare-contract-seal", command)
        run(
            "build-contract",
            [
                sys.executable,
                str(HERE / "contracts_forensic_v4.py"),
                "--seal",
                str(seal_path),
                "--output",
                str(contract_path),
            ],
        )
        run(
            "build-registration",
            [
                sys.executable,
                str(HERE / "prepare_forensic_training_registration_v4.py"),
                "--contract",
                str(contract_path),
                "--mode",
                mode,
                "--first",
                str(phase_first),
                "--deadline",
                str(deadline),
                "--cpu-core",
                str(cpu_core),
                "--output",
                str(output),
                "--registration-out",
                str(registration_path),
            ],
        )
        run(
            "native-proof-or-fit",
            [
                sys.executable,
                str(HERE / "prove.py"),
                "--registration",
                str(registration_path),
            ],
        )
        result = json.loads((output / "result.json").read_bytes())
        if result.get("status") != expected_status:
            raise ValueError("typed v4 native phase did not pass")
        record.update(
            status=(
                "PASS-forensic-v4-whole-pause-fresh-native-proof-not-strength"
                if mode == "proof"
                else "PASS-forensic-v4-fresh64-native-loads-not-strength"
            ),
            contract=ref(contract_path),
            contract_build_seal=ref(seal_path),
            training_registration=ref(registration_path),
            phase_result=ref(output / "result.json"),
            candidate=(ref(output / "whole/candidate.pt") if mode == "fresh-fit" else None),
        )
        guard()
    except BaseException as error:
        record.update(status="FAILED-preserved", error=repr(error))
        raise
    finally:
        record["finished"] = time.time()
        write_once(phase_dir / "result.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--mode", choices=("proof", "fresh-fit"), required=True)
    parser.add_argument("--cpu-core", type=int, choices=(1, 3), required=True)
    arguments = parser.parse_args()
    execute(arguments.root, arguments.seed, arguments.mode, arguments.cpu_core)
