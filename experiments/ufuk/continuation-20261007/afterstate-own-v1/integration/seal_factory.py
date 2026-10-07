"""ROOT-only seal/registration constructors; never assign clocks or launch work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

AFTERSTATE_SEAL = "NNUE-own-afterstate1024-conversion-seal-v1"
BUILD_SEAL = "own-nnue-afterstate-contract-build-seal-v1"
ORCHESTRATION = "NNUE-own-afterstate-training-orchestration-v1"
VARIANT = "afterstate-search-q-v1"
OPERATOR_END_MAX = 1791448916.685839
PROOF_STATUS = "PASS-afterstate-fixed-phase-and-fresh-native-loads-not-strength"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def check_tree(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            path = Path(value["path"])
            if not path.is_file() or sha(path) != value["sha256"]:
                raise ValueError("pinned original ownQ input changed")
        for child in value.values():
            check_tree(child)
    elif isinstance(value, list):
        for child in value:
            check_tree(child)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _phase_clock(mode, first, deadline, operator_end_epoch):
    cap = 600 if mode == "proof" else 1800
    if (
        mode not in {"proof", "fresh-fit"}
        or type(first) not in (int, float)
        or type(deadline) not in (int, float)
        or type(operator_end_epoch) not in (int, float)
        or not first < deadline <= min(first + cap, operator_end_epoch)
        or operator_end_epoch > OPERATOR_END_MAX
    ):
        raise ValueError("explicit ROOT phase clock within fixed operator window required")


def make_conversion_spec(
    source_spec_path,
    source_conversion_result_path,
    *,
    first,
    deadline,
    operator_end_epoch,
    source_converter_path,
    core_repo,
):
    """Create a new conversion seal over a completed original ownQ-v2 result.

    This helper verifies existing public metadata/files and takes all clock
    values from ROOT. It does not launch conversion or read model weights.
    """
    if type(first) not in (int, float) or type(deadline) not in (int, float):
        raise ValueError("explicit ROOT conversion clock required")
    if not first < deadline <= min(first + 600, operator_end_epoch):
        raise ValueError("afterstate conversion clock must fit one registered 600-second phase")
    if operator_end_epoch > OPERATOR_END_MAX:
        raise ValueError("operator window cannot be extended")
    source_spec_path = Path(source_spec_path).resolve()
    result_path = Path(source_conversion_result_path).resolve()
    source_spec = json.loads(source_spec_path.read_bytes())
    check_tree(source_spec)
    result = json.loads(result_path.read_bytes())
    if (
        source_spec.get("schema") != "NNUE-own1024-dataset-conversion-seal-v2"
        or result.get("status") != "PASS-own1024-fullhistory-trace-conversion-not-strength"
        or result.get("dataset_sha256") is None
        or result.get("provenance_sha256") is None
    ):
        raise ValueError("completed frozen ownQ-v2 conversion required")
    data_path = result_path.parent / "dataset.json"
    provenance_path = result_path.parent / "provenance.json"
    if (
        sha(data_path) != result["dataset_sha256"]
        or sha(provenance_path) != result["provenance_sha256"]
    ):
        raise ValueError("original ownQ-v2 result/data/provenance bytes differ")
    source_provenance = json.loads(provenance_path.read_bytes())
    registration_ref, receipt_ref = source_spec["registration"], source_spec["receipt"]
    registration = json.loads(Path(registration_ref["path"]).read_bytes())
    receipt = json.loads(Path(receipt_ref["path"]).read_bytes())
    if (
        sha(registration_ref["path"]) != registration_ref["sha256"]
        or sha(receipt_ref["path"]) != receipt_ref["sha256"]
        or registration.get("schema") != "own-nnue-ownq-collection-registration-v2"
        or receipt.get("schema") != "own-nnue-ownq-collection-receipt-v2"
        or receipt.get("status") != "PASS-exact-row-budget"
        or len(receipt.get("training_row_ids", [])) != 1024
        or source_provenance.get("collection_receipt_sha256") != receipt_ref["sha256"]
        or source_provenance.get("dataset_sha256") != result["dataset_sha256"]
    ):
        raise ValueError("same-seed successful ownQ-v2 collection/conversion lineage required")
    old_converter = Path(source_converter_path).resolve()
    if old_converter.name != "convert.py" or not old_converter.is_file():
        raise ValueError("pinned original ownQ-v2 converter required")
    parent = registration["parent_candidate"]
    if parent.get("sha256") is None or sha(parent["path"]) != parent["sha256"]:
        raise ValueError("same-seed original named teacher parent bytes required")
    spec = {
        "schema": AFTERSTATE_SEAL,
        "status": "registered",
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": operator_end_epoch,
        "seed": registration["seed"],
        "core_repo": str(Path(core_repo).resolve()),
        "source_converter": ref(old_converter),
        "source_conversion_seal": ref(source_spec_path),
        "source_conversion_result": ref(result_path),
        "source_dataset": ref(data_path),
        "source_provenance": ref(provenance_path),
        "parent_candidate": parent,
        "features": source_spec["features"],
        "prior": source_spec["prior"],
    }
    return spec


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
    _phase_clock(mode, first, deadline, operator_end_epoch)
    dataset_ref, provenance_ref = ref(dataset), ref(provenance)
    data = json.loads(Path(dataset_ref["path"]).read_bytes())
    prov = json.loads(Path(provenance_ref["path"]).read_bytes())
    parent_ref, parent_contract_ref = ref(parent_candidate), ref(parent_contract)
    parent_contract_obj = json.loads(Path(parent_contract_ref["path"]).read_bytes())
    if (
        data.get("schema") != "own-kingbucket-afterstate-search-q-data-v1"
        or len(data.get("rows", [])) != 1024
        or prov.get("schema") != "NNUE-own1024-afterstate-search-q-provenance-v1"
        or prov.get("seed") != seed
        or prov.get("dataset_sha256") != dataset_ref["sha256"]
        or prov.get("parent_candidate", {}).get("sha256") != parent_ref["sha256"]
        or parent_contract_obj.get("seed") != seed
        or parent_contract_obj.get("phase") != "teacher-bootstrap"
        or parent_contract_obj.get("updates") != 256
    ):
        raise ValueError("afterstate data, same-seed parent, and teacher contract must agree")
    if prov.get("inputs", {}).get("schema") != AFTERSTATE_SEAL:
        raise ValueError("afterstate conversion input spec required")
    for path, digest in inference_source_sha256.items():
        if sha(path) != digest:
            raise ValueError("pinned inference source differs")
    seal = {
        "schema": BUILD_SEAL,
        "status": "registered",
        "mode": mode,
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": operator_end_epoch,
        "seed": seed,
        "dataset": dataset_ref,
        "provenance": provenance_ref,
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
            raise ValueError("fresh-fit seal requires completed same-data proof refs")
        proof_ref, proof_contract_ref = ref(proof_result), ref(proof_contract)
        proof = json.loads(Path(proof_ref["path"]).read_bytes())
        proof_contract_obj = json.loads(Path(proof_contract_ref["path"]).read_bytes())
        if (
            proof.get("status") != PROOF_STATUS
            or proof.get("mode") != "proof"
            or proof.get("seed") != seed
            or proof.get("dataset_sha256") != dataset_ref["sha256"]
            or proof.get("target_provenance_sha256") != provenance_ref["sha256"]
            or proof_contract_obj.get("phase") != "own-afterstate-search-q-learning-v1"
            or proof_contract_obj.get("execution_mode") != "proof"
        ):
            raise ValueError("completed same-data/parent afterstate proof required")
        seal["proof_result"] = proof_ref
        seal["proof_contract"] = proof_contract_ref
    elif proof_result is not None or proof_contract is not None:
        raise ValueError("proof phase cannot name a previous proof")
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
    target_provenance,
    parent_candidate,
    build_seal,
    source_dir,
    output,
):
    _phase_clock(mode, first, deadline, operator_end_epoch)
    source_dir = Path(source_dir).resolve()
    c = json.loads(Path(contract).read_bytes())
    contract_ref = ref(contract)
    dataset_ref = ref(dataset)
    provenance_ref = ref(target_provenance)
    parent_ref = ref(parent_candidate)
    build_seal_ref = ref(build_seal)
    if (
        c.get("seed") != seed
        or c.get("phase") != "own-afterstate-search-q-learning-v1"
        or c.get("execution_mode") != mode
        or c.get("original_first_epoch") != first
        or c.get("original_deadline_epoch") != deadline
        or c.get("operator_end_epoch") != operator_end_epoch
        or c.get("dataset_sha256") != dataset_ref["sha256"]
        or c.get("target_provenance_sha256") != provenance_ref["sha256"]
        or c.get("parent_candidate_sha256") != parent_ref["sha256"]
        or c.get("contract_build_seal_sha256") != build_seal_ref["sha256"]
    ):
        raise ValueError("contract must bind exact ROOT clock and phase")
    source_sha256 = c["source_sha256"] | c["execution_helpers_sha256"]
    return {
        "schema": ORCHESTRATION,
        "status": "registered",
        "mode": mode,
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": operator_end_epoch,
        "seed": seed,
        "cpu_core": cpu_core,
        "train": ref(source_dir / "train.py"),
        "native": ref(source_dir / "native.py"),
        "contract": contract_ref,
        "dataset": dataset_ref,
        "target_provenance": provenance_ref,
        "parent_candidate": parent_ref,
        "source_sha256": source_sha256,
        "inputs": {
            "contract": contract_ref,
            "dataset": dataset_ref,
            "target_provenance": provenance_ref,
            "parent_candidate": parent_ref,
            "build_seal": build_seal_ref,
        },
        "output": str(Path(output).resolve()),
    }


def make_child_record(
    *,
    seed,
    contract,
    proof_contract,
    dataset,
    target_provenance,
    candidate,
    proof_result,
    fit_result,
    proof_registration,
    fit_registration,
    collection_receipt,
    collection_registration,
    events,
    proof_build_seal,
    fit_build_seal,
    variant_helpers,
):
    """Build the typed child record; all inputs must already exist and be pinned."""
    refs = {
        name: ref(path)
        for name, path in {
            "contract": contract,
            "proof_contract": proof_contract,
            "dataset": dataset,
            "target_provenance": target_provenance,
            "candidate": candidate,
            "proof_result": proof_result,
            "fit_result": fit_result,
            "proof_registration": proof_registration,
            "fit_registration": fit_registration,
            "collection_receipt": collection_receipt,
            "collection_registration": collection_registration,
            "events": events,
        }.items()
    }
    helpers = {name: ref(path) for name, path in variant_helpers.items()}
    if set(helpers) != {"model", "native", "contract", "convert", "contract_builder"}:
        raise ValueError("exact afterstate variant helper closure required")
    c = json.loads(Path(refs["contract"]["path"]).read_bytes())
    proof_contract_obj = json.loads(Path(refs["proof_contract"]["path"]).read_bytes())
    provenance = json.loads(Path(refs["target_provenance"]["path"]).read_bytes())
    proof = json.loads(Path(refs["proof_result"]["path"]).read_bytes())
    fit = json.loads(Path(refs["fit_result"]["path"]).read_bytes())
    proof_registration_obj = json.loads(Path(refs["proof_registration"]["path"]).read_bytes())
    fit_registration_obj = json.loads(Path(refs["fit_registration"]["path"]).read_bytes())
    proof_seal, fit_seal = ref(proof_build_seal), ref(fit_build_seal)
    if (
        c.get("seed") != seed
        or c.get("phase") != "own-afterstate-search-q-learning-v1"
        or c.get("execution_mode") != "fresh-fit"
        or proof.get("status") != PROOF_STATUS
        or proof.get("seed") != seed
        or proof.get("mode") != "proof"
        or fit.get("status") != PROOF_STATUS
        or fit.get("seed") != seed
        or fit.get("mode") != "fresh-fit"
        or proof.get("contract_sha256") != ref(proof_contract)["sha256"]
        or fit.get("contract_sha256") != refs["contract"]["sha256"]
        or proof.get("registration_sha256") != refs["proof_registration"]["sha256"]
        or fit.get("registration_sha256") != refs["fit_registration"]["sha256"]
        or c.get("contract_build_seal_sha256") != fit_seal["sha256"]
        or proof_contract_obj.get("contract_build_seal_sha256") != proof_seal["sha256"]
        or proof_registration_obj.get("schema") != ORCHESTRATION
        or proof_registration_obj.get("mode") != "proof"
        or fit_registration_obj.get("schema") != ORCHESTRATION
        or fit_registration_obj.get("mode") != "fresh-fit"
        or c.get("raw_collection_inputs") != provenance.get("inputs")
    ):
        raise ValueError("completed same-seed afterstate proof and fit required")
    expected_source = c["source_sha256"] | c["execution_helpers_sha256"]
    for name, helper_name in {
        "model": "model.py",
        "native": "native.py",
        "contract": "contract.py",
        "convert": "convert.py",
        "contract_builder": "contract_builder.py",
    }.items():
        if str(Path(helpers[name]["path"]).resolve()) not in expected_source:
            raise ValueError(f"{helper_name} absent from fitted contract source closure")
        if expected_source[str(Path(helpers[name]["path"]).resolve())] != helpers[name]["sha256"]:
            raise ValueError(f"{helper_name} SHA differs from fitted contract")
    source_spec = provenance["inputs"]
    source_conversion_spec = json.loads(
        Path(source_spec["source_conversion_seal"]["path"]).read_bytes()
    )
    if (
        ref(source_spec["source_conversion_seal"]["path"]) != source_spec["source_conversion_seal"]
        or source_conversion_spec.get("receipt") != refs["collection_receipt"]
        or source_conversion_spec.get("registration") != refs["collection_registration"]
        or source_conversion_spec.get("events") != refs["events"]
    ):
        raise ValueError("child receipt/registration/events must be the original ownQ-v2 inputs")
    return {
        "seed": seed,
        **refs,
        "initial": ref(fit["native_payloads"][0]["path"]),
        "native": ref(fit["native_payloads"][-1]["path"]),
        "contract_build_seals": {"proof": proof_seal, "fit": fit_seal},
        "variant_helpers": helpers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    conversion = sub.add_parser("conversion")
    conversion.add_argument("--source-spec", required=True)
    conversion.add_argument("--source-result", required=True)
    conversion.add_argument("--source-converter", required=True)
    conversion.add_argument("--core-repo", required=True)
    conversion.add_argument("--first", required=True, type=float)
    conversion.add_argument("--deadline", required=True, type=float)
    conversion.add_argument("--operator-end-epoch", required=True, type=float)
    conversion.add_argument("--output", required=True)

    build = sub.add_parser("build")
    build.add_argument("--mode", required=True, choices=["proof", "fresh-fit"])
    build.add_argument("--first", required=True, type=float)
    build.add_argument("--deadline", required=True, type=float)
    build.add_argument("--operator-end-epoch", required=True, type=float)
    build.add_argument("--seed", required=True, type=int)
    for name in [
        "dataset",
        "provenance",
        "parent-candidate",
        "parent-contract",
        "prior-helper",
        "inference-source-json",
        "core-source-repo",
        "core-source-commit",
        "feature-schema",
        "output",
    ]:
        build.add_argument("--" + name, required=True)
    build.add_argument("--proof-result")
    build.add_argument("--proof-contract")

    orchestration = sub.add_parser("orchestration")
    orchestration.add_argument("--mode", required=True, choices=["proof", "fresh-fit"])
    orchestration.add_argument("--first", required=True, type=float)
    orchestration.add_argument("--deadline", required=True, type=float)
    orchestration.add_argument("--operator-end-epoch", required=True, type=float)
    orchestration.add_argument("--seed", required=True, type=int)
    orchestration.add_argument("--cpu-core", required=True, type=int)
    for name in [
        "contract",
        "dataset",
        "target-provenance",
        "parent-candidate",
        "build-seal",
        "source-dir",
        "output-dir",
        "output",
    ]:
        orchestration.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "conversion":
        value = make_conversion_spec(
            args.source_spec,
            args.source_result,
            first=args.first,
            deadline=args.deadline,
            operator_end_epoch=args.operator_end_epoch,
            source_converter_path=args.source_converter,
            core_repo=args.core_repo,
        )
        write_once(args.output, value)
    elif args.command == "build":
        inference = json.loads(Path(args.inference_source_json).read_bytes())
        value = make_build_seal(
            mode=args.mode,
            first=args.first,
            deadline=args.deadline,
            operator_end_epoch=args.operator_end_epoch,
            seed=args.seed,
            dataset=args.dataset,
            provenance=args.provenance,
            parent_candidate=args.parent_candidate,
            parent_contract=args.parent_contract,
            feature_schema=args.feature_schema,
            prior_helper=args.prior_helper,
            inference_source_sha256=inference,
            core_source_repo=args.core_source_repo,
            core_source_commit=args.core_source_commit,
            proof_result=args.proof_result,
            proof_contract=args.proof_contract,
        )
        write_once(args.output, value)
    else:
        value = make_orchestration(
            mode=args.mode,
            first=args.first,
            deadline=args.deadline,
            operator_end_epoch=args.operator_end_epoch,
            seed=args.seed,
            cpu_core=args.cpu_core,
            contract=args.contract,
            dataset=args.dataset,
            target_provenance=args.target_provenance,
            parent_candidate=args.parent_candidate,
            build_seal=args.build_seal,
            source_dir=args.source_dir,
            output=args.output_dir,
        )
        write_once(args.output, value)


if __name__ == "__main__":
    main()
