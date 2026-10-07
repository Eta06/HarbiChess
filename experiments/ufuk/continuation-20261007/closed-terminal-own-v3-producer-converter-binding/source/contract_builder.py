"""Build a closed-terminal own-WDL training contract from a ROOT seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from contract import (
    CONTRACT_SCHEMA,
    DATA_SCHEMA,
    NATIVE_SCHEMA,
    PHASE,
    PROVENANCE_SCHEMA,
    TARGET_METHOD,
    admit_contract,
)
from native import MATH

HERE = Path(__file__).resolve().parent
SEAL_SCHEMA = "own-nnue-closed-terminal-contract-build-seal-v1"
PARENT_PHASE = "teacher-bootstrap"
OPERATOR_END = 1791448916.685839


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(ref: dict) -> Path:
    path = Path(ref["path"]).resolve()
    if sha(path) != ref["sha256"]:
        raise ValueError("pinned input SHA differs")
    return path


def check_tree(value: object) -> None:
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            pinned(value)
        for child in value.values():
            check_tree(child)
    elif isinstance(value, list):
        for child in value:
            check_tree(child)


def make_contract(seal: dict) -> dict:
    if seal.get("schema") != SEAL_SCHEMA or seal.get("status") != "registered":
        raise ValueError("ROOT-registered closed-terminal contract seal required")
    mode = seal.get("mode")
    if mode not in {"proof", "fresh-fit"}:
        raise ValueError("separate proof/fresh-fit registration required")
    first, end, operator_end = (
        seal.get("first"),
        seal.get("deadline"),
        seal.get("operator_end_epoch"),
    )
    cap = 600 if mode == "proof" else 1800
    if (
        any(type(x) not in (int, float) or not math.isfinite(x) for x in (first, end, operator_end))
        or not first < end <= min(first + cap, operator_end)
        or operator_end > OPERATOR_END
    ):
        raise ValueError("immutable phase clock/operator ceiling required")
    dataset_path = pinned(seal["dataset"])
    provenance_path = pinned(seal["provenance"])
    dataset_bytes, provenance_bytes = dataset_path.read_bytes(), provenance_path.read_bytes()
    data, provenance = json.loads(dataset_bytes), json.loads(provenance_bytes)
    if (
        data.get("schema") != DATA_SCHEMA
        or data.get("phase") != PHASE
        or data.get("target_method") != TARGET_METHOD
        or len(data.get("rows", [])) != 1024
        or provenance.get("schema") != PROVENANCE_SCHEMA
        or provenance.get("target_method") != TARGET_METHOD
        or provenance.get("output_train_rows") != 1024
        or provenance.get("seed") != seal.get("seed")
        or provenance.get("teacher_labels_used") is not False
        or provenance.get("unknown_rows_excluded") is not True
    ):
        raise ValueError("exact 1,024-row closed-terminal dataset/provenance required")
    expected = {
        "source_dataset": provenance.get("dataset_sha256"),
        "source_provenance": seal.get("provenance", {}).get("sha256"),
        "collection_receipt": provenance.get("collection_receipt_sha256"),
        "events": provenance.get("inputs", {}).get("events", {}).get("sha256"),
    }
    if any(seal.get("source_inputs", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("registered converted dataset/collection lineage differs")
    parent = pinned(seal["parent_candidate"])
    parent_contract_path = pinned(seal["parent_contract"])
    parent_contract = json.loads(parent_contract_path.read_bytes())
    if (
        parent_contract.get("phase") != PARENT_PHASE
        or parent_contract.get("updates") != 256
        or parent_contract.get("seed") != seal.get("seed")
        or parent_contract.get("feature_schema") != seal.get("feature_schema")
        or provenance.get("parent_candidate", {}).get("sha256") != sha(parent)
        or Path(provenance["parent_candidate"].get("path", "")).resolve() != parent
    ):
        raise ValueError("exact same-seed frozen teacher parent required")
    check_tree(seal.get("raw_collection_inputs"))
    inference = seal.get("inference_source_sha256")
    if not isinstance(inference, dict) or not inference:
        raise ValueError("complete portable inference source closure required")
    for path, digest in inference.items():
        if sha(path) != digest:
            raise ValueError("inference source SHA differs")
    source_names = ["model.py", "native.py", "train.py"]
    helper_names = [
        "collector.py",
        "convert.py",
        "contract.py",
        "contract_builder.py",
        "prove.py",
        "arena_adapter.py",
    ]
    source_sha = {str((HERE / name).resolve()): sha(HERE / name) for name in source_names}
    helper_sha = {str((HERE / name).resolve()): sha(HERE / name) for name in helper_names}
    result = {
        "schema": CONTRACT_SCHEMA,
        "phase": PHASE,
        "native_schema": NATIVE_SCHEMA,
        "dataset_schema": DATA_SCHEMA,
        "target_method": TARGET_METHOD,
        "updates": 64,
        "seed": seal["seed"],
        "math": MATH,
        "feature_schema": seal["feature_schema"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha(dataset_path),
        "target_provenance_path": str(provenance_path),
        "target_provenance_sha256": sha(provenance_path),
        "parent_candidate_path": str(parent),
        "parent_candidate_sha256": sha(parent),
        "initializer_kind": "named-parent-weights-only",
        "optimizer_mode": "fresh-adam",
        "initial_accepted_step": 0,
        "legacy_native_resume_allowed": False,
        "bootstrap_candidate_path": str(parent),
        "bootstrap_candidate_sha256": sha(parent),
        "prior_helper_path": str(pinned(seal["prior_helper"])),
        "prior_helper_sha256": seal["prior_helper"]["sha256"],
        "inference_source_sha256": inference,
        "source_sha256": source_sha,
        "execution_helpers_sha256": helper_sha,
        "execution_scope_schema": "NNUE-own-execution-contract-v2",
        "execution_mode": mode,
        "original_first_epoch": first,
        "original_deadline_epoch": end,
        "operator_end_epoch": operator_end,
        "core_source_repo": seal["core_source_repo"],
        "core_source_commit": seal["core_source_commit"],
        "raw_collection_inputs": seal["raw_collection_inputs"],
        "collection_receipt_sha256": provenance["collection_receipt_sha256"],
        "teacher_labels_used_in_own_phase": False,
        "weights_bridge": "named teacher MODEL weights only; fresh Adam/global/private sampler RNG",
        "contract_build_seal_sha256": seal.get("seal_sha256"),
    }
    if mode == "fresh-fit":
        proof_path = pinned(seal["proof_result"])
        proof_contract_path = pinned(seal["proof_contract"])
        proof = json.loads(proof_path.read_bytes())
        if (
            proof.get("status")
            != "PASS-closed-terminal-fixed-phase-and-fresh-native-loads-not-strength"
            or proof.get("mode") != "proof"
            or proof.get("own_updates") != 8
            or proof.get("full_payload_bits_equal") is not True
            or proof.get("weights_only_initializer", {}).get("sha256") != sha(parent)
            or proof.get("dataset_sha256") != sha(dataset_path)
            or proof.get("target_provenance_sha256") != sha(provenance_path)
            or proof.get("seed") != seal["seed"]
            or proof.get("contract_sha256") != sha(proof_contract_path)
            or [x.get("step") for x in proof.get("native_payloads", [])] != [0, 8, 0, 4, 4, 8]
            or any(
                x.get("actual_fresh_process") is not True for x in proof.get("native_payloads", [])
            )
        ):
            raise ValueError("same-data/parent closed-terminal proof required before fit")
        if not proof.get("first") < proof.get("finished") < proof.get("deadline"):
            raise ValueError("proof must finish inside original clock")
        for payload in proof["native_payloads"]:
            pinned(payload)
        proof_contract = json.loads(proof_contract_path.read_bytes())
        if (
            proof_contract.get("phase") != PHASE
            or proof_contract.get("execution_mode") != "proof"
            or proof_contract.get("dataset_sha256") != sha(dataset_path)
            or proof_contract.get("target_provenance_sha256") != sha(provenance_path)
            or proof_contract.get("parent_candidate_sha256") != sha(parent)
        ):
            raise ValueError("proof contract must bind same data/parent")
        result["closed_terminal_proof_result"] = seal["proof_result"]
    admit_contract(canonical(result), dataset_bytes, provenance_bytes)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raw = a.seal.read_bytes()
    seal = json.loads(raw)
    seal["seal_sha256"] = hashlib.sha256(raw).hexdigest()
    contract = make_contract(seal)
    if not a.output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM contract publication only")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("xb") as stream:
        stream.write(canonical(contract) + b"\n")


if __name__ == "__main__":
    main()
