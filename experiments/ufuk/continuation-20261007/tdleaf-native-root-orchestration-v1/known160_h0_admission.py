"""H0-only native admission for a future TDLeaf known-development arena.

This checks source, full native, conversion and actual phase receipts. It does
not search, run games, call the evaluator, or claim a strength result.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

STAGE = Path(
    "/workspace/work/harbichess/continuation-20261007/"
    "tdleaf-human-prior-own-v2-native-integration-v1"
).resolve()
ORIGINAL = Path(
    "/workspace/work/harbichess/continuation-20261007/"
    "tdleaf-human-prior-own-v2-qualified-producer-v3"
).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(262144), b""):
            h.update(block)
    return h.hexdigest()


def pin(ref, cap=16 * 2**20):
    if set(ref) != {"path", "sha256"}:
        raise ValueError("exact path/SHA reference required")
    path = Path(ref["path"])
    if path.is_symlink():
        raise ValueError("pinned regular file/cap/SHA mismatch")
    path = path.resolve(strict=True)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > cap
        or sha(path) != ref["sha256"]
    ):
        raise ValueError("pinned regular file/cap/SHA mismatch")
    return path


def read(ref):
    return json.loads(pin(ref).read_bytes())


def load_module(ref, name, directory=None):
    path = pin(ref)
    old_path = list(sys.path)
    if directory:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def _phase(bundle, seed, mode, proof_contract=None):
    if set(bundle) != {"controller_result", "contract", "seal", "registration", "result"}:
        raise ValueError("exact controller/contract/seal/registration/result bundle required")
    outer = read(bundle["controller_result"])
    contract = read(bundle["contract"])
    seal = read(bundle["seal"])
    registration = read(bundle["registration"])
    result = read(bundle["result"])
    expected_updates = 8 if mode == "proof" else 64
    if (
        outer.get("schema") != "tdleaf-native-root-orchestration-phase-result-v1"
        or outer.get("status") != "PASS-phase-executed-not-strength"
        or outer.get("seed") != seed
        or outer.get("mode") != mode
        or outer.get("contract") != bundle["contract"]
        or outer.get("phase_result") != bundle["result"]
        or contract.get("phase") != "human-prior-tdleaf-own-learning-v2"
        or contract.get("seed") != seed
        or contract.get("updates") != 64
        or contract.get("tdleaf_lambda") != 0.5
        or contract.get("teacher_labels_used_in_own_phase") is not False
        or contract.get("execution_mode") != mode
        or contract.get("contract_build_seal") != seal
        or result.get("status")
        != "PASS-TDLeaf-fixed-phase-and-fresh-native-loads-not-strength"
        or result.get("mode") != mode
        or result.get("own_updates") != expected_updates
        or result.get("registration_sha256") != bundle["registration"]["sha256"]
        or result.get("contract_sha256") != bundle["contract"]["sha256"]
        or result.get("dataset_sha256") != contract.get("dataset_sha256")
        or result.get("target_provenance_sha256") != contract.get("target_provenance_sha256")
        or result.get("teacher_labels_used") is not False
        or result.get("raw_zip_identity_claimed") is not False
        or not result.get("first", 0) < result.get("finished", 0) <= result.get("deadline", 0)
        or result.get("deadline", 0) - result.get("first", 0) > (600 if mode == "proof" else 1800)
        or registration.get("schema") != "human-prior-tdleaf-training-orchestration-v2"
        or registration.get("seed") != seed
        or registration.get("mode") != mode
        or registration.get("contract") != bundle["contract"]
        or registration.get("contract_build_seal") != bundle["seal"]
        or registration.get("dataset", {}).get("sha256") != contract.get("dataset_sha256")
        or registration.get("target_provenance", {}).get("sha256")
        != contract.get("target_provenance_sha256")
        or seal.get("schema") != "human-prior-tdleaf-contract-build-seal-v2"
        or seal.get("seed") != seed
        or seal.get("mode") != mode
        or seal.get("dataset", {}).get("sha256") != contract.get("dataset_sha256")
        or seal.get("target_provenance", {}).get("sha256")
        != contract.get("target_provenance_sha256")
    ):
        raise ValueError("phase receipt/contract/clock/registration mismatch")
    if proof_contract is not None and (
        contract.get("proof_contract") != proof_contract
        or seal.get("proof_contract") != proof_contract
        or result.get("full_payload_bits_equal") is not True
    ):
        raise ValueError("fresh64 must bind its exact same-seed whole/pause/resume proof")
    return outer, contract, seal, registration, result


def _check_native_rows(native, phase, contract, registration, contract_ref, steps):
    if [row.get("step") for row in phase["native_payloads"]] != steps:
        raise ValueError("complete strict native payload steps required")
    states = []
    commands = phase.get("commands", [])
    if len(commands) != (9 if len(steps) == 6 else 3):
        raise ValueError("actual 9 proof or 3 fresh-fit owned child commands required")
    for row, step in zip(phase["native_payloads"], steps, strict=True):
        native_path = pin(row)
        if row.get("actual_fresh_process") is not True:
            raise ValueError("actual independent strict native process required")
        matches = []
        for owner in commands:
            argv = owner.get("command", [])
            if (
                "--audit-only" in argv
                and "--resume" in argv
                and argv[argv.index("--resume") + 1] == str(native_path)
            ):
                matches.append(owner)
        if len(matches) != 1:
            raise ValueError("one registered strict fresh-process load per native")
        owner = matches[0]
        argv = owner["command"]
        log = pin({"path": row["log_path"], "sha256": owner["log_sha256"]})
        if (
            owner.get("returncode") != 0
            or owner.get("finished", 0) > phase.get("deadline", float("inf"))
            or argv[1] != registration["train"]["path"]
            or argv[argv.index("--contract") + 1] != contract_ref["path"]
            or argv[argv.index("--dataset") + 1] != registration["dataset"]["path"]
            or argv[argv.index("--resume-sha256") + 1] != row["sha256"]
            or int(argv[argv.index("--stop") + 1]) != step
            or json.loads(log.read_bytes())
            != {"status": "PASS-strict-native-readonly", "step": step}
        ):
            raise ValueError("strict audit command/log/contract/counter binding differs")
        state = native.load_native(native_path, contract).native()
        states.append(state)
    if len(steps) == 6 and not all(
        native.bits_equal(states[a], states[b]) for a, b in ((0, 2), (1, 5), (3, 4))
    ):
        raise ValueError("whole8/pause4/resume8 complete model/Adam/RNG bits differ")
    return states


def _h0_parent(contract, collection_registration, torch):
    child_parent = contract.get("parent_candidate", {})
    collection_parent = collection_registration.get("parent_candidate", {})
    if (
        child_parent.get("path") != collection_parent.get("path")
        or child_parent.get("sha256") != collection_parent.get("sha256")
        or contract.get("parent_admission_seal")
        != collection_registration.get("parent_admission_seal")
        or contract.get("parent_admission_result")
        != collection_registration.get("parent_admission_result")
        or contract.get("generation") != 1
        or contract.get("parent_generation") != 0
    ):
        raise ValueError("TDLeaf must derive directly from the admitted original H0 parent")
    paths = [
        p for p in contract.get("execution_helpers_sha256", {}) if p.endswith("/parent_bridge.py")
    ]
    if len(paths) != 1:
        raise ValueError("exact original H0 parent bridge closure required")
    bridge_ref = {"path": paths[0], "sha256": contract["execution_helpers_sha256"][paths[0]]}
    bridge = load_module(bridge_ref, "tdleaf_known160_original_h0_bridge", STAGE)
    parent_spec, parent_contract = bridge.validate_admission_result(
        contract["parent_admission_seal"], contract["parent_admission_result"]
    )
    if (
        parent_spec.get("seed") != contract["seed"]
        or parent_spec.get("generation") != 1
        or parent_contract.get("seed") != contract["seed"]
        or parent_contract.get("generation") != 0
        or parent_contract.get("phase") != "human-prior-zero-residual-init-v1"
        or parent_contract.get("updates") != 0
        or parent_contract.get("teacher_labels_used_in_own_phase") is not False
    ):
        raise ValueError("literal H0 zero parent, never teacher256")
    native_ref = parent_spec["parent_native_helper"]
    model_ref = parent_spec["parent_model_helper"]
    parent_dir = Path(model_ref["path"]).resolve().parent
    parent_native = load_module(native_ref, "tdleaf_known160_h0_native", parent_dir)
    parent_result, admission_result = bridge.admit(
        parent_spec,
        parent_native,
        lambda path: torch.load(path, map_location="cpu", weights_only=False),
    )
    state = parent_native.load_native(pin(parent_spec["parent_native"]), parent_contract).native()
    packet = torch.load(
        pin(parent_spec["parent_candidate"]), map_location="cpu", weights_only=False
    )
    if (
        packet.get("contract") != parent_contract
        or packet.get("schema") != parent_native.MODEL_SCHEMA
        or not parent_native.bits_equal(packet.get("model"), parent_result)
        or state.get("step") != 0
        or not parent_native.bits_equal(state["model"], parent_result)
        or not admission_result.get("incoming_native_full_resume_checked")
    ):
        raise ValueError("literal-zero H0 candidate/full native storage mismatch")
    return parent_result, parent_spec["parent_candidate"]


def _verify_collection_and_targets(spec, collection, receipt, contract, fit_registration):
    inputs = contract.get("raw_collection_inputs")
    if not isinstance(inputs, dict):
        raise ValueError("nested exact raw collection inputs required")
    expected = {
        "registration": spec["collection_registration"],
        "receipt": spec["collection_receipt"],
        "events": spec["events"],
    }
    for key, value in expected.items():
        if inputs.get(key) != value:
            raise ValueError("contract raw collection refs differ from actual closed v4 source")
    if inputs.get("producer_directory") != str(ORIGINAL):
        raise ValueError("original frozen actor helper directory required")
    provenance_ref = {
        "path": contract.get("target_provenance_path"),
        "sha256": contract.get("target_provenance_sha256"),
    }
    dataset_ref = fit_registration.get("dataset")
    provenance = read(provenance_ref)
    dataset_path = pin(dataset_ref, 8 * 2**20)
    if (
        provenance.get("schema") != "human-prior-tdleaf-target-provenance-v2"
        or provenance.get("status") != "PASS-replayed-closed-PV-leaf-targets"
        or provenance.get("inputs") != inputs
        or provenance.get("receipt_sha256") != spec["collection_receipt"]["sha256"]
        or provenance.get("registration_sha256") != spec["collection_registration"]["sha256"]
        or sha(dataset_path) != contract.get("dataset_sha256")
    ):
        raise ValueError("dataset/provenance refs and raw v4 source disagree")
    sys.path.insert(0, str(STAGE))
    converter = load_module(
        {"path": str(STAGE / "convert.py"), "sha256": sha(STAGE / "convert.py")},
        "tdleaf_known160_converter",
        STAGE,
    )
    try:
        rebuilt_dataset, rebuilt_provenance = converter.convert_sealed(provenance["inputs"])
    finally:
        sys.path.pop(0)
    if (
        converter.canonical(rebuilt_dataset) + b"\n" != dataset_path.read_bytes()
        or converter.canonical(rebuilt_provenance) + b"\n" != pin(provenance_ref).read_bytes()
    ):
        raise ValueError("independent full TDLeaf conversion replay differs")
    audit = read(contract["collection_audit_result"])
    clock = read(contract["collection_audit_clock"])
    audit_helper = pin(contract["collection_audit_helper"])
    if (
        audit.get("schema") != "human-prior-tdleaf-six-root-pv-audit-result-v2"
        or audit.get("status")
        != "PASS-six-actual-chronological-TDLeaf-PV-packets-and-full-alias-traces"
        or audit.get("seed") != spec["seed"]
        or audit.get("helper_sha256") != sha(audit_helper)
        or audit.get("registration_sha256") != spec["collection_registration"]["sha256"]
        or audit.get("receipt_sha256") != spec["collection_receipt"]["sha256"]
        or audit.get("events_sha256") != spec["events"]["sha256"]
        or audit.get("clock_sha256") != contract["collection_audit_clock"]["sha256"]
        or [row.get("row_id") for row in audit.get("packets", [])]
        != receipt.get("periodic_independent_search_rows")
        or len(audit.get("packets", [])) != 6
        or any(audit.get(k) != 0 for k in ("new_training_rows", "new_games", "optimizer_updates"))
        or clock.get("schema") != "human-prior-tdleaf-six-root-audit-clock-v2"
        or clock.get("helper_sha256") != sha(audit_helper)
        or clock.get("registration_sha256") != spec["collection_registration"]["sha256"]
        or clock.get("receipt_sha256") != spec["collection_receipt"]["sha256"]
        or clock.get("events_sha256") != spec["events"]["sha256"]
        or clock.get("first") != audit.get("first")
        or clock.get("deadline") != audit.get("deadline")
        or not audit.get("first", 0) < audit.get("finished", 0) <= audit.get("deadline", 0)
    ):
        raise ValueError("actual six chronological PV/protection audit required")


def admit_child(spec):
    if (
        spec.get("schema") != "tdleaf-own-v2-h0-known160-child-v1"
        or spec.get("status") != "registered"
        or spec.get("seed") not in (20262905, 20262906)
        or "teachers" in spec
    ):
        raise ValueError("typed TDLeaf H0 child only; no legacy or teacher phase alias")
    import torch

    seed = spec["seed"]
    collection = read(spec["collection_registration"])
    root_result = read(spec["root_collection_result"])
    receipt = read(spec["collection_receipt"])
    if (
        root_result.get("status")
        != "PASS-actual1024-TDLeaf-current-parent-selfplay-not-strength"
        or root_result.get("collection_registration") != spec["collection_registration"]
        or root_result.get("raw_receipt") != spec["collection_receipt"]
        or collection.get("seed") != seed
        or receipt.get("seed") != seed
        or receipt.get("train_rows") != 1024
        or receipt.get("teacher_labels_used") is not False
    ):
        raise ValueError("actual original sealed closed TDLeaf1024 actor source required")
    proof_bundle, fit_bundle = spec["proof"], spec["fresh_fit"]
    proof_outer, proof_contract, proof_seal, proof_registration, proof = _phase(
        proof_bundle, seed, "proof"
    )
    fit_outer, contract, fit_seal, fit_registration, fit = _phase(
        fit_bundle, seed, "fresh-fit", proof_bundle["contract"]
    )
    for field in (
        "dataset_sha256",
        "target_provenance_sha256",
        "raw_collection_inputs",
        "search_helper",
        "pv_search_helper",
        "parent_candidate_sha256",
        "parent_admission_seal",
        "parent_admission_result",
        "math",
        "feature_schema",
    ):
        if proof_contract.get(field) != contract.get(field):
            raise ValueError(
                "proof/fresh-fit must preserve identical data, search, H0 and objective"
            )
    if (
        contract.get("bootstrap_candidate_sha256") != collection["parent_candidate"]["sha256"]
        or contract.get("bootstrap_candidate_path") != collection["parent_candidate"]["path"]
        or contract.get("parent_admission_seal") != collection["parent_admission_seal"]
        or contract.get("parent_admission_result") != collection["parent_admission_result"]
        or contract.get("teacher_labels_used_in_own_phase") is not False
        or contract.get("generation") != 1
        or "teachers" in spec
    ):
        raise ValueError("same H0-only new Adam/RNG weights initializer; teacher role forbidden")
    if spec.get("models") != {
        "parent": collection["parent_candidate"],
        "learned": fit_bundle["candidate"],
    }:
        raise ValueError("known160 parent/candidate mapping must preserve exact H0 and child")
    _verify_collection_and_targets(spec, collection, receipt, contract, fit_registration)

    # Re-derive each immutable contract from its own source seal at recorded time.
    sys.path.insert(0, str(STAGE))
    builder = load_module(
        {"path": str(STAGE / "contracts.py"), "sha256": sha(STAGE / "contracts.py")},
        "tdleaf_known160_contract_builder",
        STAGE,
    )
    original_time = time.time
    try:
        for phase_seal, expected in (
            (proof_seal, proof_contract),
            (fit_seal, contract),
        ):
            time.time = lambda s=phase_seal: (s["first"] + s["deadline"]) / 2
            if builder.build(phase_seal) != expected:
                raise ValueError("TDLeaf contract does not rederive from immutable phase seal")
    finally:
        time.time = original_time
        sys.path.pop(0)

    native_ref = fit_registration["native"]
    native = load_module(native_ref, "tdleaf_known160_child_native", STAGE)
    for source_map in (contract["source_sha256"], contract["execution_helpers_sha256"]):
        for path, digest in source_map.items():
            if sha(path) != digest:
                raise ValueError("TDLeaf training helper/source closure changed")
    proof_states = _check_native_rows(
        native,
        proof,
        proof_contract,
        proof_registration,
        proof_bundle["contract"],
        [0, 8, 0, 4, 4, 8],
    )
    fit_states = _check_native_rows(
        native,
        fit,
        contract,
        fit_registration,
        fit_bundle["contract"],
        [0, 64],
    )
    if not native.bits_equal(proof_states[0]["model"], proof_states[1]["baseline"]):
        raise ValueError("proof baseline must be the initialized H0 storage")
    parent_model, parent_candidate_ref = _h0_parent(contract, collection, torch)
    zero, final = fit_states
    derived_candidate = {
        "path": str((Path(fit_registration["output"]) / "whole/candidate.pt").resolve()),
        "sha256": sha(Path(fit_registration["output"]) / "whole/candidate.pt"),
    }
    if fit_bundle.get("candidate") != derived_candidate:
        raise ValueError("candidate must be the registered fresh-fit whole/fixed64 output")
    candidate = torch.load(pin(fit_bundle["candidate"]), map_location="cpu", weights_only=False)
    if (
        zero["step"] != 0
        or zero["optimizer"]["state"]
        or not native.bits_equal(zero["model"], parent_model)
        or not native.bits_equal(zero["baseline"], parent_model)
        or not native.bits_equal(final["baseline"], parent_model)
        or native.bits_equal(final["model"], parent_model)
        or candidate.get("schema") != native.MODEL_SCHEMA
        or candidate.get("contract") != contract
        or not native.bits_equal(candidate.get("model"), final["model"])
    ):
        raise ValueError("fresh64 must change the exact H0 model under new native state")
    return {
        "schema": "tdleaf-own-v2-h0-known160-child-admission-v1",
        "status": "PASS-actual-H0-parent-and-TDLeaf-native-not-strength",
        "seed": seed,
        "parent": parent_candidate_ref,
        "candidate": fit_bundle["candidate"],
        "contract": fit_bundle["contract"],
        "proof": proof_bundle["result"],
        "fit": fit_bundle["result"],
        "full_native_checks": 8,
        "teacher256_role": False,
        "strength_claim": False,
    }
