"""Explicit integration-only v2 registration projection into unchanged proof owner."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PUBLIC = "human-randomstarts-own-training-orchestration-v2"
EXECUTOR = "human-prior-own-training-orchestration-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(ref):
    if set(ref) != {"path", "sha256"} or sha(ref["path"]) != ref["sha256"]:
        raise ValueError("exact immutable adapter input/source reference")
    return Path(ref["path"]).resolve()


def project(registration_ref, proof_ref):
    path, proof = pin(registration_ref), pin(proof_ref)
    original = json.loads(path.read_bytes())
    if (
        original.get("schema") != PUBLIC
        or original.get("status") != "registered"
        or original.get("mode") not in {"proof", "fresh-fit"}
        or "registration_sha256" in original
    ):
        raise ValueError(
            "only original public procedural registration accepts projection"
        )
    sources = original["source_sha256"]
    if (
        proof.name != "prove.py"
        or sources.get(str(proof)) != proof_ref["sha256"]
        or pin(original["train"]).parent != proof.parent
        or pin(original["native"]).parent != proof.parent
        or pin(original["metadata_spec_helper"]) != proof.with_name("specs.py")
    ):
        raise ValueError("unchanged proof and exact same frozen procedural source tree")
    for source, digest in sources.items():
        pin({"path": source, "sha256": digest})
    converter = proof.with_name("convert.py")
    if sources.get(str(converter)) != sha(converter):
        raise ValueError("unchanged imported converter helper source")
    validate_phase_closure(original, proof)
    projected = dict(
        original, schema=EXECUTOR, registration_sha256=registration_ref["sha256"]
    )
    return original, projected, proof, converter


def validate_phase_closure(reg, proof):
    c = json.loads(pin(reg["contract"]).read_bytes())
    for source, digest in c["execution_helpers_sha256"].items():
        pin({"path": source, "sha256": digest})
    if (
        c["execution_helpers_sha256"].get(str(proof)) != sha(proof)
        or c["execution_helpers_sha256"].get(str(proof.with_name("specs.py")))
        != reg["metadata_spec_helper"]["sha256"]
        or c["phase"] != "own-learning"
        or c["updates"] != 64
        or c["root_source_type"] != "procedural-uniform-legal-walk-v2"
        or c["execution_scope_schema"] != "human-prior-own-execution-contract-v1"
        or c["execution_mode"] != reg["mode"]
        or c["seed"] != reg["seed"]
        or (
            c["original_first_epoch"],
            c["original_deadline_epoch"],
            c["operator_end_epoch"],
        )
        != (reg["first"], reg["deadline"], reg["operator_end_epoch"])
        or c["dataset_sha256"] != reg["dataset"]["sha256"]
        or sha(pin(reg["dataset"])) != c["dataset_sha256"]
    ):
        raise ValueError("exact procedural phase/source/data/original clock closure")
    bridge = proof.with_name("parent_bridge.py")
    if c["execution_helpers_sha256"].get(str(bridge)) != sha(bridge):
        raise ValueError("exact current parent admission bridge source")
    spec = importlib.util.spec_from_file_location(
        "procedural_adapter_parent_metadata", bridge
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent_seal, parent_contract = module.validate_admission_result(
        c["parent_admission_seal"], c["parent_admission_result"]
    )
    parent_binding = dict(
        parent_seal["parent_candidate"],
        contract_sha256=hashlib.sha256(
            json.dumps(
                parent_contract, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
    )
    if (
        reg["parent_candidate"] != parent_binding
        or c["parent_candidate"] != parent_seal["parent_candidate"]
        or c["generation"] != parent_seal["generation"]
        or c["parent_generation"] != parent_contract["generation"]
        or c["generation"] != parent_contract["generation"] + 1
        or c["bootstrap_candidate_path"] != parent_binding["path"]
        or c["bootstrap_candidate_sha256"] != parent_binding["sha256"]
        or c["search_helper"] != parent_contract["search_helper"]
        or c["teacher_labels_used_in_own_phase"] is not False
    ):
        raise ValueError("exact named current parent/canonical contract/generation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "proof-helper", "derivation-receipt"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--registration-sha256", required=True)
    p.add_argument("--proof-helper-sha256", required=True)
    a = p.parse_args()
    registration_ref = {
        "path": str(a.registration.resolve()),
        "sha256": a.registration_sha256,
    }
    proof_ref = {"path": str(a.proof_helper.resolve()), "sha256": a.proof_helper_sha256}
    original, projected, proof, converter = project(registration_ref, proof_ref)
    cap = 600 if original["mode"] == "proof" else 1800
    if not (
        original["first"]
        <= time.time()
        < original["deadline"]
        <= min(
            original["first"] + cap, original["operator_end_epoch"], 1791448916.685839
        )
    ):
        raise ValueError("existing original proof/fit clock; no reset")
    if not a.derivation_receipt.resolve().is_relative_to("/dev/shm"):
        raise ValueError("publish-once RAM adapter receipt")
    receipt = dict(
        schema="procedural-proof-registration-projection-v1",
        status="registered-explicit-integration-projection-not-proof-PASS",
        original_registration=registration_ref,
        original_public_schema=PUBLIC,
        executor_registration_schema=EXECUTOR,
        original_proof_helper=proof_ref,
        adapter={"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        changed_original_fields=["schema"],
        injected_executor_field="registration_sha256=original-public-file-SHA",
        first=original["first"],
        deadline=original["deadline"],
        observed_epoch=time.time(),
    )
    a.derivation_receipt.parent.mkdir(parents=True, exist_ok=True)
    with a.derivation_receipt.open("xb") as stream:
        stream.write(json.dumps(receipt, sort_keys=True).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    sys.path.insert(0, str(proof.parent))
    for name in ("convert",):
        if name in sys.modules:
            raise ValueError(
                "fresh adapter process required, no ambiguous helper import"
            )
    spec = importlib.util.spec_from_file_location("original_procedural_proof", proof)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(sys.modules["convert"].__file__).resolve() != converter:
        raise ValueError("actual imported converter origin")
    module.execute(projected)


if __name__ == "__main__":
    main()
