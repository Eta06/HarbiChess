"""ROOT-only constructors for action-ranking conversion and training seals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

OPERATOR_END_MAX = 1791448916.685839
CONVERSION = "NNUE-own-action-ranking1024-conversion-seal-v1"
BUILD = "own-nnue-action-ranking-contract-build-seal-v1"
ORCHESTRATION = "NNUE-own-action-ranking-training-orchestration-v1"
FIT_STATUS = "PASS-action-ranking-fixed-phase-and-fresh-native-loads-not-strength"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def make_conversion_seal(
    *,
    afterstate_seal,
    afterstate_result,
    afterstate_converter,
    first,
    deadline,
    operator_end_epoch,
    seed,
):
    if (
        not first < deadline <= min(first + 600, operator_end_epoch)
        or operator_end_epoch > OPERATOR_END_MAX
    ):
        raise ValueError("explicit original conversion clock required")
    seal_path, result_path = Path(afterstate_seal).resolve(), Path(afterstate_result).resolve()
    seal = json.loads(seal_path.read_bytes())
    result = json.loads(result_path.read_bytes())
    if (
        seal.get("schema") != "NNUE-own-afterstate1024-conversion-seal-v1"
        or seal.get("status") != "registered"
        or seal.get("seed") != seed
        or result.get("status") != "PASS-afterstate-ownQ-fullhistory-conversion-not-strength"
        or not result.get("dataset_sha256")
        or not result.get("provenance_sha256")
    ):
        raise ValueError("same-seed byte-verified afterstate conversion required")
    directory = result_path.parent
    dataset, provenance = directory / "dataset.json", directory / "provenance.json"
    if sha(dataset) != result["dataset_sha256"] or sha(provenance) != result["provenance_sha256"]:
        raise ValueError("afterstate output bytes differ from result receipt")
    ownq_spec = json.loads(Path(seal["source_conversion_seal"]["path"]).read_bytes())
    if sha(seal["source_conversion_seal"]["path"]) != seal["source_conversion_seal"]["sha256"]:
        raise ValueError("pinned original ownQ conversion seal changed")
    registration = json.loads(Path(ownq_spec["registration"]["path"]).read_bytes())
    if sha(ownq_spec["registration"]["path"]) != ownq_spec["registration"]["sha256"]:
        raise ValueError("pinned original ownQ registration changed")
    parent = seal.get("parent_candidate", {})
    registered_parent = registration.get("parent_candidate", {})
    if (
        registration.get("schema") != "own-nnue-ownq-collection-registration-v2"
        or registration.get("status") != "registered"
        or registration.get("seed") != seed
        or parent.get("sha256") != registered_parent.get("sha256")
        or Path(parent.get("path", "")).resolve()
        != Path(registered_parent.get("path", "")).resolve()
    ):
        raise ValueError("same-seed original ownQ registration and parent required")
    return {
        "schema": CONVERSION,
        "status": "registered",
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": operator_end_epoch,
        "seed": seed,
        "afterstate_conversion_seal": ref(seal_path),
        "afterstate_conversion_result": ref(result_path),
        "afterstate_dataset": ref(dataset),
        "afterstate_provenance": ref(provenance),
        "afterstate_converter": ref(afterstate_converter),
        "core_source_repo": seal["core_repo"],
        "core_source_commit": registration["core_commit"],
    }


def make_build_seal(
    *,
    mode,
    first,
    deadline,
    operator_end_epoch,
    seed,
    dataset,
    provenance,
    parent_candidate,
    parent_contract,
    feature_schema,
    prior_helper,
    inference_source_sha256,
    core_source_repo,
    core_source_commit,
    proof_result=None,
    proof_contract=None,
):
    cap = 600 if mode == "proof" else 1800
    if mode not in {"proof", "fresh-fit"} or not first < deadline <= min(
        first + cap, operator_end_epoch, OPERATOR_END_MAX
    ):
        raise ValueError("ROOT must register the immutable mode clock")
    data_ref, prov_ref = ref(dataset), ref(provenance)
    parent_ref, parent_contract_ref = ref(parent_candidate), ref(parent_contract)
    data, prov = (
        json.loads(Path(data_ref["path"]).read_bytes()),
        json.loads(Path(prov_ref["path"]).read_bytes()),
    )
    parent = json.loads(Path(parent_contract_ref["path"]).read_bytes())
    if (
        data.get("schema") != "own-kingbucket-afterstate-action-ranking-data-v1"
        or len(data.get("rows", [])) != 1024
        or prov.get("schema") != "NNUE-own1024-afterstate-action-ranking-provenance-v1"
        or prov.get("seed") != seed
        or prov.get("dataset_sha256") != data_ref["sha256"]
        or prov.get("parent_candidate", {}).get("sha256") != parent_ref["sha256"]
        or parent.get("phase") != "teacher-bootstrap"
        or parent.get("updates") != 256
        or parent.get("seed") != seed
    ):
        raise ValueError("exact same-seed own ranking dataset and teacher parent required")
    seal = {
        "schema": BUILD,
        "status": "registered",
        "mode": mode,
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": operator_end_epoch,
        "seed": seed,
        "dataset": data_ref,
        "provenance": prov_ref,
        "parent_candidate": parent_ref,
        "parent_contract": parent_contract_ref,
        "feature_schema": feature_schema,
        "prior_helper": ref(prior_helper),
        "inference_source_sha256": inference_source_sha256,
        "core_source_repo": str(Path(core_source_repo).resolve()),
        "core_source_commit": core_source_commit,
    }
    if mode == "fresh-fit":
        if proof_result is None or proof_contract is None:
            raise ValueError("fresh-fit requires a completed same-data proof")
        proof_ref, proof_contract_ref = ref(proof_result), ref(proof_contract)
        proof_obj = json.loads(Path(proof_ref["path"]).read_bytes())
        proof_c = json.loads(Path(proof_contract_ref["path"]).read_bytes())
        if (
            proof_obj.get("status") != FIT_STATUS
            or proof_obj.get("mode") != "proof"
            or proof_obj.get("seed") != seed
            or proof_obj.get("dataset_sha256") != data_ref["sha256"]
            or proof_obj.get("target_provenance_sha256") != prov_ref["sha256"]
            or proof_c.get("phase") != "own-afterstate-action-ranking-v1"
            or proof_c.get("execution_mode") != "proof"
        ):
            raise ValueError("completed exact-data proof required")
        seal.update(proof_result=proof_ref, proof_contract=proof_contract_ref)
    elif proof_result is not None or proof_contract is not None:
        raise ValueError("proof cannot depend on an earlier proof")
    return seal


def make_orchestration(
    *,
    mode,
    first,
    deadline,
    operator_end_epoch,
    seed,
    cpu_core,
    contract,
    dataset,
    provenance,
    parent_candidate,
    build_seal,
    source_dir,
    output,
):
    cap = 600 if mode == "proof" else 1800
    if mode not in {"proof", "fresh-fit"} or not first < deadline <= min(
        first + cap, operator_end_epoch, OPERATOR_END_MAX
    ):
        raise ValueError("ROOT must register separate phase clocks")
    contract_ref, dataset_ref = ref(contract), ref(dataset)
    prov_ref, parent_ref, seal_ref = ref(provenance), ref(parent_candidate), ref(build_seal)
    c = json.loads(Path(contract_ref["path"]).read_bytes())
    source_dir = Path(source_dir).resolve()
    expected_sources = c["source_sha256"] | c["execution_helpers_sha256"]
    train_path = source_dir / "train.py"
    native_path = source_dir / "native.py"
    if (
        c.get("phase") != "own-afterstate-action-ranking-v1"
        or c.get("seed") != seed
        or c.get("execution_mode") != mode
        or c.get("original_first_epoch") != first
        or c.get("original_deadline_epoch") != deadline
        or c.get("operator_end_epoch") != operator_end_epoch
        or c.get("dataset_sha256") != dataset_ref["sha256"]
        or c.get("target_provenance_sha256") != prov_ref["sha256"]
        or c.get("parent_candidate_sha256") != parent_ref["sha256"]
        or c.get("contract_build_seal_sha256") != seal_ref["sha256"]
        or str(train_path) not in c["source_sha256"]
        or str(native_path) not in c["source_sha256"]
        or c["source_sha256"] | c["execution_helpers_sha256"] != expected_sources
        or any(sha(path) != digest for path, digest in expected_sources.items())
        or not Path(output).resolve().is_relative_to("/dev/shm")
    ):
        raise ValueError("contract/source/data/clock closure differs")
    return {
        "schema": "NNUE-own-action-ranking-training-orchestration-v1",
        "status": "registered",
        "mode": mode,
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": operator_end_epoch,
        "seed": seed,
        "cpu_core": cpu_core,
        "contract": contract_ref,
        "dataset": dataset_ref,
        "target_provenance": prov_ref,
        "parent_candidate": parent_ref,
        "build_seal": seal_ref,
        "source_dir": str(source_dir),
        "train": ref(train_path),
        "native": ref(native_path),
        "output": str(Path(output).resolve()),
        "source_sha256": expected_sources,
        "inputs": {
            "contract": contract_ref,
            "dataset": dataset_ref,
            "target_provenance": prov_ref,
            "parent_candidate": parent_ref,
            "build_seal": seal_ref,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("conversion", "build", "orchestration"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    makers = {
        "conversion": make_conversion_seal,
        "build": make_build_seal,
        "orchestration": make_orchestration,
    }
    result = makers[args.kind](**request)
    digest = write_once(args.output, result)
    print(json.dumps({"path": str(args.output.resolve()), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
