"""Deterministic ROOT seal construction; never assigns clocks or starts work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEAL_SCHEMA = "own-nnue-closed-terminal-contract-build-seal-v1"
# This integration correction deliberately retains the successful v3 source tree.
V3_ROOT = (
    Path(__file__).resolve().parent.parent / "closed-terminal-own-v3-producer-converter-binding"
)
SOURCE = V3_ROOT / "source"



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def make_seal(
    *,
    mode,
    first,
    deadline,
    operator_end_epoch,
    seed,
    conversion_spec,
    conversion_result,
    parent_contract,
    core_source_repo,
    core_source_commit,
    feature_schema,
    proof_result=None,
    proof_contract=None,
):
    """Create a seal only from already-produced, explicitly pinned inputs.

    Clock values and source identity are mandatory caller inputs from ROOT; this
    helper does not read wall time, select a parent, or launch any process.
    """
    if mode not in {"proof", "fresh-fit"}:
        raise ValueError("mode must be proof or fresh-fit")
    spec_path = Path(conversion_spec).resolve()
    result_path = Path(conversion_result).resolve()
    spec = json.loads(spec_path.read_bytes())
    result = json.loads(result_path.read_bytes())
    if (
        spec.get("schema") != "NNUE-closedterminal1024-conversion-seal-v1"
        or result.get("status")
        != "PASS-own1024-closed-terminal-fullhistory-conversion-not-strength"
        or result.get("dataset_sha256") is None
        or result.get("provenance_sha256") is None
    ):
        raise ValueError("completed, same-seed full converter result required")
    converter_ref = spec.get("converter_helper", {})
    expected_converter = SOURCE / "convert.py"
    if (
        spec.get("converter_binding_schema") != "closed-terminal-corrective-converter-binding-v3"
        or set(converter_ref) != {"path", "sha256"}
        or Path(converter_ref["path"]).resolve() != expected_converter
        or sha(expected_converter) != converter_ref["sha256"]
    ):
        raise ValueError("corrective conversion/build source binding required")
    dataset_ref = ref(result_path.parent / "dataset.json")
    provenance_ref = ref(result_path.parent / "provenance.json")
    if (
        dataset_ref["sha256"] != result["dataset_sha256"]
        or provenance_ref["sha256"] != result["provenance_sha256"]
    ):
        raise ValueError("converter result/output bytes differ")
    provenance = json.loads(Path(provenance_ref["path"]).read_bytes())
    registration_ref = spec["registration"]
    if sha(registration_ref["path"]) != registration_ref["sha256"]:
        raise ValueError("converter registration ref no longer matches")
    if sha(spec["receipt"]["path"]) != spec["receipt"]["sha256"]:
        raise ValueError("converter receipt ref no longer matches")
    registration = json.loads(Path(registration_ref["path"]).read_bytes())
    if (
        registration.get("seed") != seed
        or provenance.get("collection_receipt_sha256") != spec["receipt"]["sha256"]
        or provenance.get("dataset_sha256") != dataset_ref["sha256"]
    ):
        raise ValueError("converted files do not bind the registered MC collection")
    parent = registration["parent_candidate"]
    parent_contract_ref = ref(parent_contract)
    parent_contract_obj = json.loads(Path(parent_contract_ref["path"]).read_bytes())
    if parent_contract_obj.get("seed") != seed:
        raise ValueError("same-seed parent contract required")
    parent_helper = {
        "path": registration["parent_helpers"]["prior_path"],
        "sha256": registration["parent_helpers"]["prior_sha256"],
    }
    raw = dict(spec)
    raw["conversion_spec_file"] = ref(spec_path)
    raw["conversion_result"] = ref(result_path)
    raw["parent_contract"] = parent_contract_ref
    source_inputs = {
        "source_dataset": dataset_ref["sha256"],
        "source_provenance": provenance_ref["sha256"],
        "collection_receipt": spec["receipt"]["sha256"],
        "events": spec["events"]["sha256"],
    }
    helper_dir = Path(registration["parent_helpers"]["directory"]).resolve()
    parent_helpers = registration["parent_helpers"]
    inference_refs = {
        str(helper_dir / "model.py"): parent_helpers["model_sha256"],
        str(helper_dir / "native.py"): parent_helpers["native_sha256"],
        str(helper_dir / "evaluator.py"): parent_helpers["evaluator_sha256"],
        parent_helpers["extension_path"]: parent_helpers["extension_sha256"],
    }
    for path, digest in inference_refs.items():
        if sha(path) != digest:
            raise ValueError("parent inference source differs")
    seal = {
        "schema": SEAL_SCHEMA,
        "status": "registered",
        "mode": mode,
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": operator_end_epoch,
        "seed": seed,
        "dataset": dataset_ref,
        "provenance": provenance_ref,
        "source_inputs": source_inputs,
        "parent_candidate": parent,
        "parent_contract": parent_contract_ref,
        "prior_helper": parent_helper,
        "feature_schema": feature_schema,
        "inference_source_sha256": inference_refs,
        "raw_collection_inputs": raw,
        "core_source_repo": str(Path(core_source_repo).resolve()),
        "core_source_commit": core_source_commit,
    }
    if mode == "fresh-fit":
        if proof_result is None or proof_contract is None:
            raise ValueError("fresh-fit seal requires completed proof refs")
        seal["proof_result"] = ref(proof_result)
        seal["proof_contract"] = ref(proof_contract)
    elif proof_result is not None or proof_contract is not None:
        raise ValueError("proof mode does not accept prior proof refs")
    return seal


def write_once(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def make_conversion_spec(
    registration_path, receipt_path, *, first, deadline, core_repo,
    producer_directory, converter_helper,
):
    """Construct the converter seal from already frozen collector artifacts."""
    registration_path = Path(registration_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    reg = json.loads(registration_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    if (
        reg.get("schema") != "own-nnue-closed-terminal-collection-registration-v1"
        or reg.get("status") != "registered"
        or receipt.get("schema") != "own-nnue-closed-terminal-collection-receipt-v1"
        or receipt.get("status") != "PASS-exact-closed-terminal-row-budget"
        or receipt.get("registration_sha256") != sha(registration_path)
        or receipt.get("seed") != reg.get("seed")
    ):
        raise ValueError("successful, same-seed immutable collection required")
    output = Path(reg["output_path"]).resolve()
    if Path(receipt_path).resolve() != output / "receipt.json":
        raise ValueError("receipt must come from registered producer output")
    helper_dir = Path(reg["parent_helpers"]["directory"]).resolve()
    event_path = output / "events.jsonl"
    if (
        event_path.stat().st_size != receipt["events_bytes"]
        or sha(event_path) != receipt["events_sha256"]
    ):
        raise ValueError("collector event stream differs from receipt")
    producer_dir = Path(producer_directory).resolve()
    converter_ref = dict(converter_helper)
    if set(converter_ref) != {"path", "sha256"}:
        raise ValueError("explicit corrective converter ref required")
    converter_path = Path(converter_ref["path"]).resolve()
    expected_converter = SOURCE / "convert.py"
    if converter_path != expected_converter or sha(converter_path) != converter_ref["sha256"]:
        raise ValueError("exact new corrective converter source required")
    converter_ref["path"] = str(converter_path)
    if reg["producer_source_sha256"] != receipt["producer_source_sha256"]:
        raise ValueError("immutable recorded producer identities differ")
    for name, digest in reg["producer_source_sha256"].items():
        path = (producer_dir / name).resolve()
        if path.parent != producer_dir or sha(path) != digest:
            raise ValueError("actual immutable producer directory source differs")
    chunks = {}
    for item in receipt["alias_chunks"]:
        path = (output / item["file"]).resolve()
        if (
            path.parent != output
            or path.stat().st_size != item["bytes"]
            or sha(path) != item["sha256"]
        ):
            raise ValueError("collector alias chunk differs from receipt")
        chunks[item["file"]] = ref(path)
    return {
        "schema": "NNUE-closedterminal1024-conversion-seal-v1",
        "status": "registered",
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": reg["operator_end_epoch"],
        "core_repo": str(Path(core_repo).resolve()),
        "registration": ref(registration_path),
        "receipt": ref(receipt_path),
        "producer_directory": str(producer_dir),
        "converter_binding_schema": "closed-terminal-corrective-converter-binding-v3",
        "converter_helper": converter_ref,
        "features": ref(helper_dir / "model.py"),
        "prior": ref(reg["parent_helpers"]["prior_path"]),
        "events": ref(event_path),
        "alias_chunks": chunks,
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
    """Build a typed known160 MC child record from completed files only."""
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
    c = json.loads(Path(refs["contract"]["path"]).read_bytes())
    proof_contract_obj = json.loads(Path(refs["proof_contract"]["path"]).read_bytes())
    proof = json.loads(Path(refs["proof_result"]["path"]).read_bytes())
    fit = json.loads(Path(refs["fit_result"]["path"]).read_bytes())
    proof_seal = ref(proof_build_seal)
    fit_seal = ref(fit_build_seal)
    helpers = {name: ref(path) for name, path in variant_helpers.items()}
    if set(helpers) != {"model", "native", "contract", "convert", "contract_builder"}:
        raise ValueError("exact MC variant helper closure required")
    if (
        c.get("seed") != seed
        or c.get("phase") != "own-closed-terminal-learning-v1"
        or c.get("execution_mode") != "fresh-fit"
        or c.get("contract_build_seal_sha256") != fit_seal["sha256"]
        or proof_contract_obj.get("contract_build_seal_sha256") != proof_seal["sha256"]
        or proof.get("contract_sha256") != refs["proof_contract"]["sha256"]
        or fit.get("contract_sha256") != refs["contract"]["sha256"]
        or proof.get("registration_sha256") != refs["proof_registration"]["sha256"]
        or fit.get("registration_sha256") != refs["fit_registration"]["sha256"]
        or proof.get("status")
        != "PASS-closed-terminal-fixed-phase-and-fresh-native-loads-not-strength"
        or proof.get("mode") != "proof"
        or proof.get("seed") != seed
        or proof.get("own_updates") != 8
        or [x.get("step") for x in proof.get("native_payloads", [])] != [0, 8, 0, 4, 4, 8]
        or fit.get("status")
        != "PASS-closed-terminal-fixed-phase-and-fresh-native-loads-not-strength"
        or fit.get("mode") != "fresh-fit"
        or fit.get("seed") != seed
        or fit.get("own_updates") != 64
        or [x.get("step") for x in fit.get("native_payloads", [])] != [0, 64]
    ):
        raise ValueError("same-seed proof and fixed fresh64 result required")
    for key in ("proof_result", "fit_result"):
        for payload in json.loads(Path(refs[key]["path"]).read_bytes())["native_payloads"]:
            if payload.get("actual_fresh_process") is not True:
                raise ValueError("actual strict fresh-process native loads required")
            if sha(payload["path"]) != payload["sha256"]:
                raise ValueError("native payload reference changed")
    closure = {**c.get("source_sha256", {}), **c.get("execution_helpers_sha256", {})}
    if any(
        helper["path"] not in closure or closure[helper["path"]] != helper["sha256"]
        for helper in helpers.values()
    ):
        raise ValueError("MC variant helper refs must be in the frozen contract source closure")
    return {
        **refs,
        "variant_helpers": helpers,
        "initial": ref(fit["native_payloads"][0]["path"]),
        "native": ref(fit["native_payloads"][1]["path"]),
        "contract_build_seals": {"proof": proof_seal, "fit": fit_seal},
    }


def make_audit_set(entries):
    """Create MC-only two-seed audit set; old Q/TD audit schemas are rejected."""
    required_seeds = {20262905, 20262906}
    if {entry["seed"] for entry in entries} != required_seeds or len(entries) != 2:
        raise ValueError("exact two-seed MC audit entries required")
    audits = []
    for entry in sorted(entries, key=lambda item: item["seed"]):
        result_ref, clock_ref = ref(entry["result"]), ref(entry["clock"])
        result = json.loads(Path(result_ref["path"]).read_bytes())
        if (
            result.get("schema") != "NNUE-own-closed-terminal-six-search-audit-v1"
            or result.get("status")
            != (
                "PASS-six-actual-chronological-terminal-eligible-search-packets-"
                "and-complete-alias-traces"
            )
            or result.get("seed") != entry["seed"]
            or result.get("clock_sha256") != clock_ref["sha256"]
            or len(result.get("packets", [])) != 6
            or result.get("new_games") != 0
            or result.get("new_training_rows") != 0
            or result.get("optimizer_updates") != 0
        ):
            raise ValueError("actual MC six-packet audit result required")
        audits.append({"seed": entry["seed"], "result": result_ref, "clock": clock_ref})
    return {
        "schema": "NNUE-own-closed-terminal-replay-audit-set-v1",
        "helper": ref(V3_ROOT / "integration" / "audit_six.py"),
        "audits": audits,
    }
