"""ROOT-clocked own64 contracts from admitted zero/current-own parent."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from convert_forensic_v3 import canonical, module, pinned, pins_tree, sha

HERE = Path(__file__).resolve().parent


def clock(spec):
    first, end, operator = spec["first"], spec["deadline"], spec["operator_end_epoch"]
    cap = 600 if spec["mode"] == "proof" else 1800
    if (
        spec["mode"] not in ("proof", "fresh-fit")
        or any(type(x) not in (float, int) or not math.isfinite(x) for x in (first, end, operator))
        or not first < end <= min(first + cap, operator)
        or operator > 1791448916.685839
    ):
        raise ValueError("ROOT new prospective phase clocks; no historical clock mutation")


def build(spec):
    if (
        spec["schema"] != "human-randomstarts-own-forensic-contract-build-seal-v3"
        or spec["status"] != "registered"
    ):
        raise ValueError("ROOT immutable contract build seal")
    clock(spec)
    from forensic_audit_set import validate as validate_forensic_audits

    audit_set = json.loads(pinned(spec["forensic_audit_set"]).read_bytes())
    if (
        spec["forensic_audit_validator"] != audit_set["validator"]
        or spec["forensic_audit_runtime"] != audit_set["helper"]
    ):
        raise ValueError("exact v3 audit validator")
    pinned(spec["forensic_audit_validator"])
    validate_forensic_audits(audit_set)
    data = pinned(spec["dataset"])
    provenance_path = pinned(spec["provenance"])
    provenance = json.loads(provenance_path.read_bytes())
    rows = json.loads(data.read_bytes())
    if (
        rows["schema"] != "own-kingbucket-sparse-training-data-forensic-v3"
        or rows["phase"] != "own-learning"
        or len(rows["rows"]) != 1024
        or provenance["schema"] != "human-randomstarts-own1024-forensic-data-provenance-v3"
        or provenance["teacher_labels_used"] is not False
        or provenance["dataset_sha256"] != spec["dataset"]["sha256"]
        or provenance["seed"] != spec["seed"]
    ):
        raise ValueError("strict1024 own-data and converted provenance")
    if (
        provenance.get("forensic_audit_set", {}).get("path")
        != str(Path(spec["forensic_audit_set"]["path"]).resolve())
        or provenance.get("forensic_audit_set", {}).get("sha256")
        != spec["forensic_audit_set"]["sha256"]
        or provenance.get("forensic_audit_set", {}).get("validator")
        != spec["forensic_audit_validator"]
        or provenance.get("forensic_audit_set", {}).get("runtime") != spec["forensic_audit_runtime"]
    ):
        raise ValueError("forensic audit set is part of converted-data provenance")
    pins_tree(provenance["inputs"])
    conversion = provenance["inputs"]
    receipt = json.loads(pinned(conversion["receipt"]).read_bytes())
    if (
        provenance.get("raw_receipt_registration_sha256") != receipt.get("registration_sha256")
        or provenance.get("verified_original_registration_sha256")
        != conversion["registration"]["sha256"]
        or provenance.get("forensic_view", {}).get("raw_receipt_bytes_preserved") is not True
    ):
        raise ValueError("forensic corrected-binding provenance without raw receipt rewrite")
    from parent_bridge import validate_admission_result

    parent_seal, parent_contract = validate_admission_result(
        spec["parent_admission_seal"], spec["parent_admission_result"]
    )
    if receipt["parent_candidate_sha256"] != parent_seal["parent_candidate"]["sha256"]:
        raise ValueError("exact current own-parent produces own targets")
    sys.path.insert(0, str(HERE))
    native = module(
        dict(path=str(HERE / "native.py"), sha256=sha(HERE / "native.py")),
        "own_contract_native",
    )
    collection_reg = json.loads(pinned(conversion["registration"]).read_bytes())
    parent_binding = collection_reg["parent_candidate"]
    expected_parent_contract_sha = hashlib.sha256(canonical(parent_contract)).hexdigest()
    if (
        parent_binding["contract_sha256"] != expected_parent_contract_sha
        or receipt["parent_contract_sha256"] != expected_parent_contract_sha
        or parent_binding["path"] != parent_seal["parent_candidate"]["path"]
        or collection_reg["generation"] != parent_seal["generation"]
        or collection_reg["parent_admission_result"] != spec["parent_admission_result"]
    ):
        raise ValueError("same full own-parent contract produces own targets")
    if collection_reg["search_helper"] != parent_contract["search_helper"]:
        raise ValueError("parent/child SAME fixed search, no search-only gain labeled learning")
    if spec["seed"] != parent_contract["seed"]:
        raise ValueError("same named lineage seed")
    inference = dict(parent_contract["inference_source_sha256"])
    for path, h in inference.items():
        pinned(dict(path=path, sha256=h))
    own_proof = None
    if spec["mode"] == "fresh-fit":
        own_proof = json.loads(pinned(spec["own_proof_result"]).read_bytes())
        if (
            own_proof["status"] != "PASS-own-NNUE-fixed-phase-and-fresh-native-loads-not-strength"
            or own_proof["mode"] != "proof"
            or own_proof["own_updates"] != 8
            or own_proof["full_payload_bits_equal"] is not True
            or own_proof["weights_only_initializer"]["sha256"]
            != parent_seal["parent_candidate"]["sha256"]
            or [x["step"] for x in own_proof["native_payloads"]] != [0, 8, 0, 4, 4, 8]
            or not own_proof["first"] < own_proof["finished"] < own_proof["deadline"]
        ):
            raise ValueError("actual own proof before production admission")
        for item in own_proof["native_payloads"]:
            pinned(item)
            log = Path(item["log_path"])
            if json.loads(log.read_bytes()) != dict(
                status="PASS-strict-native-readonly", step=item["step"]
            ):
                raise ValueError("six actual own fresh-load results")
        proof_contract = json.loads(pinned(spec["own_proof_contract"]).read_bytes())
        if (
            own_proof["contract_sha256"] != spec["own_proof_contract"]["sha256"]
            or proof_contract["dataset_sha256"] != sha(data)
            or proof_contract["bootstrap_candidate_sha256"]
            != parent_seal["parent_candidate"]["sha256"]
            or proof_contract["seed"] != spec["seed"]
            or proof_contract["math"] != native.MATH
        ):
            raise ValueError("same-data same-parent actual own proof")
        inference[spec["own_proof_result"]["path"]] = spec["own_proof_result"]["sha256"]
    source = {str(HERE / n): sha(HERE / n) for n in ("model.py", "native.py", "train.py")}
    helpers = {
        str(HERE / n): sha(HERE / n)
        for n in (
            "convert_forensic_v3.py",
            "contracts_forensic_v3.py",
            "prove.py",
            "parent_bridge.py",
            "zero_parent.py",
            "initialize.py",
            "admit_parent.py",
            "specs.py",
            "audit_collection_six_v3.py",
            "forensic_receipt.py",
            "forensic_audit_set.py",
            "root_bank.py",
        )
    }
    return dict(
        phase="own-learning",
        root_source_type="procedural-uniform-legal-walk-v2",
        lineage_origin=parent_contract["lineage_origin"],
        search_helper=parent_contract["search_helper"],
        updates=64,
        seed=spec["seed"],
        math=native.MATH,
        feature_schema=parent_contract["feature_schema"],
        dataset_sha256=sha(data),
        source_sha256=source,
        execution_helpers_sha256=helpers,
        execution_scope_schema="human-prior-own-forensic-execution-contract-v3",
        generation=parent_seal["generation"],
        parent_generation=parent_seal["generation"] - 1,
        execution_mode=spec["mode"],
        original_first_epoch=spec["first"],
        original_deadline_epoch=spec["deadline"],
        operator_end_epoch=spec["operator_end_epoch"],
        core_source_repo=parent_contract["core_source_repo"],
        core_source_commit=parent_contract["core_source_commit"],
        target_provenance_path=str(provenance_path),
        target_provenance_sha256=sha(provenance_path),
        prior_helper_path=parent_contract["prior_helper_path"],
        prior_helper_sha256=parent_contract["prior_helper_sha256"],
        inference_source_sha256=inference,
        bootstrap_candidate_path=str(pinned(parent_seal["parent_candidate"]).resolve()),
        bootstrap_candidate_sha256=parent_seal["parent_candidate"]["sha256"],
        weights_bridge="CURRENT OWN PARENT MODEL only; NEW Adam/Python/Torch/private sampler RNG",
        parent_candidate=parent_seal["parent_candidate"],
        parent_native=parent_seal["parent_native"],
        parent_admission_seal=spec["parent_admission_seal"],
        parent_admission_result=spec["parent_admission_result"],
        parent_ancestry=parent_seal,
        teacher_labels_used_in_own_phase=False,
        own_phase_proof=spec.get("own_proof_result"),
        collection_receipt_sha256=conversion["receipt"]["sha256"],
        forensic_audit_set=spec["forensic_audit_set"],
        raw_collection_inputs=conversion,
        contract_build_seal=spec,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    c = build(json.loads(a.seal.read_bytes()))
    if not a.output.resolve().is_relative_to("/dev/shm"):
        raise ValueError("RAM contract publish only")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("xb") as f:
        f.write(canonical(c) + b"\n")
