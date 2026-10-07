"""Prepare a v3 own-proof/fresh-fit contract seal after full forensic replay."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from forensic_audit_set import validate
from forensic_receipt import canonical, read, ref

END = 1791448916.685839


def build(
    root: Path,
    seed: int,
    mode: str,
    first: float,
    deadline: float,
    dataset: Path,
    provenance: Path,
    audit_set_path: Path,
    validator: Path,
    proof_result: Path | None = None,
    proof_contract: Path | None = None,
) -> dict:
    if seed not in (20262905, 20262906) or mode not in ("proof", "fresh-fit"):
        raise ValueError("fixed seed and v3 phase")
    cap = 600 if mode == "proof" else 1800
    registration = read(
        root / f"human-random-collect-actual-{seed}" / "collection-registration.json"
    )
    if (
        any(not math.isfinite(x) for x in (first, deadline))
        or not first
        <= time.time()
        < deadline
        <= min(first + cap, registration["operator_end_epoch"])
        or registration["operator_end_epoch"] > END
    ):
        raise ValueError("ROOT-observed original phase clock")
    manifest = json.loads(audit_set_path.read_bytes())
    validate(manifest)
    data_ref, prov_ref, audit_ref, validator_ref = map(
        ref, (dataset, provenance, audit_set_path, validator)
    )
    prov = read(provenance)
    audit_item = next((x for x in manifest["audits"] if x["seed"] == seed), None)
    if (
        audit_item is None
        or prov.get("schema") != "human-randomstarts-own1024-forensic-data-provenance-v3"
        or prov.get("seed") != seed
        or prov.get("dataset_sha256") != data_ref["sha256"]
        or prov.get("forensic_audit_set", {}).get("path") != audit_ref["path"]
        or prov.get("forensic_audit_set", {}).get("sha256") != audit_ref["sha256"]
        or manifest.get("validator") != validator_ref
    ):
        raise ValueError("same-seed converted dataset and both forensic audits")
    seal = {
        "schema": "human-randomstarts-own-forensic-contract-build-seal-v4",
        "status": "registered",
        "seed": seed,
        "mode": mode,
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": registration["operator_end_epoch"],
        "dataset": data_ref,
        "provenance": prov_ref,
        "forensic_audit_set": audit_ref,
        "forensic_audit_validator": validator_ref,
        "forensic_audit_runtime": manifest["helper"],
        "parent_admission_seal": registration["parent_admission_seal"],
        "parent_admission_result": registration["parent_admission_result"],
    }
    if mode == "fresh-fit":
        if proof_result is None or proof_contract is None:
            raise ValueError("same-data actual fresh forensic proof required")
        seal["own_proof_result"] = ref(proof_result)
        seal["own_proof_contract"] = ref(proof_contract)
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("proof", "fresh-fit"), required=True)
    parser.add_argument("--first", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--audit-set", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--proof-result", type=Path)
    parser.add_argument("--proof-contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal = build(
        args.root,
        args.seed,
        args.mode,
        args.first,
        args.deadline,
        args.dataset,
        args.provenance,
        args.audit_set,
        args.validator,
        args.proof_result,
        args.proof_contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(seal) + b"\n")
