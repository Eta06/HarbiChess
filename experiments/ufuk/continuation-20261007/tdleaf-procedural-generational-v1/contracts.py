"""Typed TDLeaf proof/fresh-fit contract builder; metadata only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import native
from parent_bridge import pinned, validate_admission_result

MAX_END = 1791448916.685839


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def build(spec):
    if (
        spec.get("schema") != "procedural-generational-tdleaf-contract-build-seal-v2"
        or spec.get("status") != "registered"
        or spec.get("mode") not in {"proof", "fresh-fit"}
    ):
        raise ValueError("ROOT-registered TDLeaf phase seal required")
    first, deadline, operator_end = (
        spec["first"],
        spec["deadline"],
        spec["operator_end_epoch"],
    )
    cap = 600 if spec["mode"] == "proof" else 1800
    if (
        any(
            type(x) not in (int, float) or not math.isfinite(x)
            for x in (first, deadline, operator_end)
        )
        or not first < deadline <= min(first + cap, operator_end)
        or operator_end > MAX_END
        or not first <= __import__("time").time() < deadline
    ):
        raise ValueError("new immutable ROOT phase clock required")
    from generation_plan import require_generation

    parent_seal, parent_contract = validate_admission_result(
        spec["parent_admission_seal"], spec["parent_admission_result"]
    )
    collection_reg = json.loads(pinned(spec["collection_registration"]).read_bytes())
    require_generation(collection_reg)
    receipt = json.loads(pinned(spec["collection_receipt"]).read_bytes())
    provenance = json.loads(pinned(spec["target_provenance"]).read_bytes())
    audit = json.loads(pinned(spec["collection_audit_result"]).read_bytes())
    audit_clock = json.loads(pinned(spec["collection_audit_clock"]).read_bytes())
    audit_helper = pinned(spec["collection_audit_helper"])
    audit_ids = receipt.get("periodic_independent_search_rows")
    dataset_path = pinned(spec["dataset"])
    dataset = json.loads(dataset_path.read_bytes())
    if (
        collection_reg.get("schema")
        != "procedural-generational-tdleaf-collection-registration-v2"
        or collection_reg.get("status") != "registered"
        or collection_reg.get("protection_scope_schema")
        != "procedural-generational-tdleaf-current-root-query-path-v2"
        or receipt.get("schema")
        != "procedural-generational-tdleaf-collection-receipt-v2"
        or receipt.get("status") != "PASS-exact-row-budget"
        or receipt.get("seed") != parent_contract["seed"]
        or collection_reg.get("seed") != parent_contract["seed"]
        or collection_reg.get("parent_admission_result")
        != spec["parent_admission_result"]
        or receipt.get("registration_sha256")
        != spec["collection_registration"]["sha256"]
        or receipt.get("teacher_labels_used") is not False
        or receipt.get("protection_scope_schema")
        != "procedural-generational-tdleaf-current-root-query-path-v2"
        or receipt.get("lambda") != 0.5
        or provenance.get("schema")
        != "procedural-generational-tdleaf-target-provenance-v2"
        or provenance.get("status") != "PASS-replayed-closed-PV-leaf-targets"
        or provenance.get("receipt_sha256") != spec["collection_receipt"]["sha256"]
        or provenance.get("registration_sha256")
        != spec["collection_registration"]["sha256"]
        or provenance.get("teacher_labels_used") is not False
        or provenance.get("protection_scope_schema")
        != "procedural-generational-tdleaf-current-root-query-path-v2"
        or provenance.get("lambda") != 0.5
        or provenance.get("converter_path")
        != str(Path(__file__).resolve().with_name("convert.py"))
        or provenance.get("converter_sha256")
        != sha(Path(__file__).with_name("convert.py"))
        or provenance.get("dataset_rows") != len(dataset.get("rows", []))
        or audit.get("schema")
        != "procedural-generational-tdleaf-six-root-pv-audit-result-v2"
        or audit.get("protection_scope_schema")
        != "procedural-generational-tdleaf-current-root-query-path-v2"
        or audit.get("status")
        != "PASS-six-actual-chronological-TDLeaf-PV-packets-and-full-alias-traces"
        or audit.get("seed") != parent_contract["seed"]
        or audit.get("helper_sha256") != sha(audit_helper)
        or audit.get("registration_sha256") != spec["collection_registration"]["sha256"]
        or audit.get("receipt_sha256") != spec["collection_receipt"]["sha256"]
        or audit.get("events_sha256") != receipt.get("events_sha256")
        or audit.get("clock_sha256") != spec["collection_audit_clock"]["sha256"]
        or [row.get("row_id") for row in audit.get("packets", [])] != audit_ids
        or [len(audit.get("packets", []))] != [6]
        or any(
            audit.get(key) != 0
            for key in ("new_training_rows", "new_games", "optimizer_updates")
        )
        or audit_clock.get("schema")
        != "procedural-generational-tdleaf-six-root-audit-clock-v2"
        or audit_clock.get("helper_sha256") != sha(audit_helper)
        or audit_clock.get("registration_sha256")
        != spec["collection_registration"]["sha256"]
        or audit_clock.get("receipt_sha256") != spec["collection_receipt"]["sha256"]
        or audit_clock.get("events_sha256") != receipt.get("events_sha256")
        or audit_clock.get("first") != audit.get("first")
        or audit_clock.get("deadline") != audit.get("deadline")
        or not audit.get("first", 0)
        < audit.get("finished", 0)
        <= audit.get("deadline", 0)
        or audit.get("deadline", 0) - audit.get("first", 0) > 600
        or audit.get("deadline", 0) > operator_end
        or dataset.get("schema")
        != "own-generational-tdleaf-pv-leaf-sparse-training-data-v1"
        or dataset.get("phase") != native.PHASE
        or dataset.get("seed") != parent_contract["seed"]
        or dataset.get("teacher_labels_used") is not False
        or dataset.get("protection_scope_schema")
        != "procedural-generational-tdleaf-current-root-query-path-v2"
        or dataset.get("lambda") != 0.5
        or dataset_path.stat().st_size > 8 * 2**20
    ):
        raise ValueError("same-seed own search/TDLeaf target/data lineage required")
    if (
        collection_reg["search_helper"] != parent_contract["search_helper"]
        or collection_reg["parent_candidate"]["path"]
        != parent_seal["parent_candidate"]["path"]
        or collection_reg["parent_candidate"]["sha256"]
        != parent_seal["parent_candidate"]["sha256"]
        or collection_reg["parent_helpers"]["prior_sha256"]
        != parent_contract["prior_helper_sha256"]
    ):
        raise ValueError("same admitted current parent/prior/search required")
    if spec["mode"] == "fresh-fit":
        proof = json.loads(pinned(spec["proof_result"]).read_bytes())
        proof_contract = json.loads(pinned(spec["proof_contract"]).read_bytes())
        if (
            proof.get("status")
            != "PASS-generational-TDLeaf-fixed-phase-and-fresh-native-loads-not-strength"
            or proof.get("mode") != "proof"
            or proof.get("own_updates") != 8
            or proof.get("full_payload_bits_equal") is not True
            or proof.get("contract_sha256") != spec["proof_contract"]["sha256"]
            or proof_contract.get("dataset_sha256") != spec["dataset"]["sha256"]
            or proof_contract.get("bootstrap_candidate_sha256")
            != parent_seal["parent_candidate"]["sha256"]
            or proof_contract.get("math") != native.MATH
        ):
            raise ValueError(
                "same-data/current-parent actual TDLeaf proof required before fit"
            )
    source = {
        str(Path(__file__).with_name(name).resolve()): sha(
            Path(__file__).with_name(name)
        )
        for name in ("model.py", "native.py", "train.py")
    }
    helpers = {
        str(Path(__file__).with_name(name).resolve()): sha(
            Path(__file__).with_name(name)
        )
        for name in (
            "convert.py",
            "contracts.py",
            "prove.py",
            "parent_bridge.py",
            "collector.py",
            "run_collection.py",
            "metadata_factory.py",
            "search_pv.py",
            "tdleaf_targets.py",
            "audit_collection_six.py",
            "generation_plan.py",
            "root_bank.py",
            "zero_parent.py",
            "admit_parent.py",
            "prepare_training.py",
            "convert_cli.py",
        )
    }
    return {
        "phase": native.PHASE,
        "lineage_origin": parent_contract["lineage_origin"],
        "search_helper": parent_contract["search_helper"],
        "pv_search_helper": collection_reg["pv_search_helper"],
        "updates": 128,
        "rng_seed": parent_contract["seed"] + 1000003 * parent_seal["generation"],
        "generation_plan": collection_reg["generation_plan"],
        "procedural_bank_receipt": collection_reg["procedural_bank_receipt"],
        "seed": parent_contract["seed"],
        "math": native.MATH,
        "feature_schema": parent_contract["feature_schema"],
        "dataset_sha256": spec["dataset"]["sha256"],
        "source_sha256": source,
        "execution_helpers_sha256": helpers,
        "execution_scope_schema": "procedural-generational-tdleaf-execution-contract-v2",
        "protection_scope_schema": "procedural-generational-tdleaf-current-root-query-path-v2",
        "generation": parent_seal["generation"],
        "parent_generation": parent_seal["generation"] - 1,
        "execution_mode": spec["mode"],
        "original_first_epoch": first,
        "original_deadline_epoch": deadline,
        "operator_end_epoch": operator_end,
        "core_source_repo": parent_contract["core_source_repo"],
        "core_source_commit": parent_contract["core_source_commit"],
        "target_provenance_path": str(pinned(spec["target_provenance"]).resolve()),
        "target_provenance_sha256": spec["target_provenance"]["sha256"],
        "prior_helper_path": parent_contract["prior_helper_path"],
        "prior_helper_sha256": parent_contract["prior_helper_sha256"],
        "inference_source_sha256": parent_contract["inference_source_sha256"],
        "bootstrap_candidate_path": str(
            pinned(parent_seal["parent_candidate"]).resolve()
        ),
        "bootstrap_candidate_sha256": parent_seal["parent_candidate"]["sha256"],
        "parent_candidate": parent_seal["parent_candidate"],
        "parent_native": parent_seal["parent_native"],
        "parent_admission_seal": spec["parent_admission_seal"],
        "parent_admission_result": spec["parent_admission_result"],
        "parent_ancestry": parent_seal,
        "teacher_labels_used_in_own_phase": False,
        "tdleaf_lambda": 0.5,
        "collection_registration": spec["collection_registration"],
        "collection_receipt": spec["collection_receipt"],
        "target_provenance": spec["target_provenance"],
        "raw_collection_inputs": provenance["inputs"],
        "collection_audit_result": spec["collection_audit_result"],
        "collection_audit_clock": spec["collection_audit_clock"],
        "collection_audit_helper": spec["collection_audit_helper"],
        "proof_result": spec.get("proof_result"),
        "proof_contract": spec.get("proof_contract"),
        "contract_build_seal": spec,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = build(json.loads(args.seal.read_bytes()))
    if not args.output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("contract artifacts publish in RAM only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(contract) + b"\n")
