"""Prepare a typed forensic-v4 proof/fresh-fit registration without running it."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from forensic_receipt import canonical, ref

END = 1791448916.685839


def build(contract_path: Path, mode: str, first: float, deadline: float, cpu_core: int,
          output: Path) -> dict:
    contract_path = Path(contract_path).resolve(strict=True)
    contract = json.loads(contract_path.read_bytes())
    cap = 600 if mode == "proof" else 1800
    if (
        contract.get("phase") != "own-learning"
        or contract.get("execution_scope_schema")
        != "human-prior-own-forensic-execution-contract-v4"
        or contract.get("native_schema") != "human-prior-own-nnue16-native-cpu-forensic-v4"
        or contract.get("execution_mode") != mode
        or not all(math.isfinite(x) for x in (first, deadline))
        or not first <= time.time() < deadline <= min(first + cap, contract["operator_end_epoch"])
        or contract["operator_end_epoch"] > END
        or (contract["original_first_epoch"], contract["original_deadline_epoch"])
        != (first, deadline)
    ):
        raise ValueError("exact ROOT-clocked typed v4 contract/phase")
    helpers = Path(__file__).resolve().parent
    refs = {
        name: ref(helpers / name)
        for name in ("train.py", "native.py", "model.py", "prove.py")
    }
    if (
        contract["source_sha256"].get(refs["train.py"]["path"]) != refs["train.py"]["sha256"]
        or contract["source_sha256"].get(refs["native.py"]["path"]) != refs["native.py"]["sha256"]
        or contract["source_sha256"].get(refs["model.py"]["path"]) != refs["model.py"]["sha256"]
    ):
        raise ValueError("contract binds exact v4 train/model/native source bytes")
    dataset_ref = contract["contract_build_seal"]["dataset"]
    inputs = {
        "contract": ref(contract_path),
        "dataset": dataset_ref,
        "target_provenance": ref(Path(contract["target_provenance_path"])),
        "parent_candidate": contract["parent_candidate"],
        "parent_native": contract["parent_native"],
        "parent_admission_seal": contract["parent_admission_seal"],
        "parent_admission_result": contract["parent_admission_result"],
        "forensic_audit_set": contract["forensic_audit_set"],
    }
    return {
        "schema": "human-prior-own-forensic-training-orchestration-v4",
        "status": "registered",
        "mode": mode,
        "seed": contract["seed"],
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": contract["operator_end_epoch"],
        "cpu_core": cpu_core,
        "train": refs["train.py"],
        "native": refs["native.py"],
        "model": refs["model.py"],
        "prove": refs["prove.py"],
        "contract": ref(contract_path),
        "dataset": dataset_ref,
        "target_provenance": inputs["target_provenance"],
        "output": str(Path(output).resolve()),
        "parent_candidate": contract["parent_candidate"],
        "inputs": inputs,
        "source_sha256": {
            **contract["source_sha256"],
            **contract["execution_helpers_sha256"],
            refs["prove.py"]["path"]: refs["prove.py"]["sha256"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--mode", choices=("proof", "fresh-fit"), required=True)
    parser.add_argument("--first", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--cpu-core", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration-out", type=Path, required=True)
    args = parser.parse_args()
    registration = build(
        args.contract,
        args.mode,
        args.first,
        args.deadline,
        args.cpu_core,
        args.output,
    )
    args.registration_out.parent.mkdir(parents=True, exist_ok=True)
    with args.registration_out.open("xb") as stream:
        stream.write(canonical(registration) + b"\n")
