"""ROOT-only closed-collection -> audit-bound conversion -> proof/fresh-fit runner.

This file consumes already registered immutable inputs. It does not start actors,
searches, or training unless ROOT explicitly invokes the proof/fresh-fit phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import contracts
from convert import canonical, convert_sealed, pinned

ROOT_END = 1791448916.685839


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(262144), b""):
            h.update(block)
    return h.hexdigest()


def read(ref):
    return json.loads(pinned(ref).read_bytes())


def write_once(ref, value, cap):
    path = Path(ref["path"]).resolve()
    if not path.is_relative_to("/dev/shm"):
        raise ValueError("phase artifacts must remain in RAM")
    payload = canonical(value) + b"\n"
    if len(payload) > cap:
        raise ValueError("publish-once artifact exceeded bound")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": sha(path)}


def validate_audit(spec, collection_reg, receipt):
    result = read(spec["collection_audit_result"])
    clock = read(spec["collection_audit_clock"])
    helper = pinned(spec["collection_audit_helper"])
    if (
        result.get("schema") != "human-prior-tdleaf-six-root-pv-audit-result-v2"
        or result.get("status")
        != "PASS-six-actual-chronological-TDLeaf-PV-packets-and-full-alias-traces"
        or result.get("seed") != collection_reg["seed"]
        or result.get("helper_sha256") != sha(helper)
        or result.get("registration_sha256") != spec["collection_registration"]["sha256"]
        or result.get("receipt_sha256") != spec["collection_receipt"]["sha256"]
        or result.get("events_sha256") != spec["events"]["sha256"]
        or result.get("clock_sha256") != spec["collection_audit_clock"]["sha256"]
        or [p.get("row_id") for p in result.get("packets", [])]
        != receipt.get("periodic_independent_search_rows")
        or len(result.get("packets", [])) != 6
        or any(result.get(k) != 0 for k in ("new_training_rows", "new_games", "optimizer_updates"))
        or clock.get("schema") != "human-prior-tdleaf-six-root-audit-clock-v2"
        or clock.get("helper_sha256") != sha(helper)
        or clock.get("registration_sha256") != spec["collection_registration"]["sha256"]
        or clock.get("receipt_sha256") != spec["collection_receipt"]["sha256"]
        or clock.get("events_sha256") != spec["events"]["sha256"]
        or clock.get("first") != result.get("first")
        or clock.get("deadline") != result.get("deadline")
        or not result.get("first", 0) < result.get("finished", 0) <= result.get("deadline", 0)
        or result.get("deadline", 0) - result.get("first", 0) > 600
    ):
        raise ValueError("actual independent six-packet audit and original clock required")
    return result


def validate_collection(spec):
    reg = read(spec["collection_registration"])
    receipt = read(spec["collection_receipt"])
    root_result = read(spec["root_collection_result"])
    source_inventory = read(spec["source_inventory"])
    events = pinned(spec["events"])
    if (
        source_inventory.get("schema") != "versioned-preactor-source-inventory-v1"
        or source_inventory.get("directory")
        != str(Path(spec["producer_directory"]).resolve())
        or spec["source_inventory"]["sha256"]
        != "750b2f64e4261704138d208a5dbd96a649d1599425ca3462d196c032a3648481"
        or root_result.get("schema") != "ROOT-actual-TDLeaf-current-parent-collection-v3"
        or root_result.get("status")
        != "PASS-actual1024-TDLeaf-current-parent-selfplay-not-strength"
        or root_result.get("collection_registration") != spec["collection_registration"]
        or root_result.get("raw_receipt") != spec["collection_receipt"]
        or root_result.get("source_inventory") != spec["source_inventory"]
        or reg.get("schema") != "human-prior-tdleaf-collection-registration-v2"
        or reg.get("status") != "registered"
        or reg.get("seed") != spec.get("seed")
        or receipt.get("schema") != "human-prior-tdleaf-collection-receipt-v2"
        or receipt.get("status") != "PASS-exact-row-budget"
        or receipt.get("seed") != spec.get("seed")
        or receipt.get("train_rows") != 1024
        or receipt.get("teacher_labels_used") is not False
        or receipt.get("registration_sha256") != spec["collection_registration"]["sha256"]
        or receipt.get("events_sha256") != spec["events"]["sha256"]
        or receipt.get("events_bytes") != events.stat().st_size
        or receipt.get("generation_helper_sha256") != reg.get("generation_helper_sha256")
        or receipt.get("parent_candidate_sha256") != reg.get("parent_candidate", {}).get("sha256")
        or receipt.get("producer_source_sha256") != reg.get("producer_source_sha256")
        or receipt.get("search") != {"nodes": 8192, "qdepth": 2, "max_depth": 8}
        or receipt.get("finished_epoch", 0) > reg.get("original_deadline_epoch", 0)
        or receipt.get("operator_end_epoch") != spec.get("operator_end_epoch")
        or reg.get("operator_end_epoch") != spec.get("operator_end_epoch")
        or reg.get("protection_scope_schema")
        != "human-prior-tdleaf-current-root-query-path-v2"
        or receipt.get("protection_scope_schema")
        != "human-prior-tdleaf-current-root-query-path-v2"
    ):
        raise ValueError("only the exact closed original v4 current-H0 collection")
    if Path(events).resolve().parent != Path(reg["output_path"]).resolve():
        raise ValueError("events must be inside registered actual collection output")
    for row in receipt.get("alias_chunks", []):
        chunk = events.parent / row["file"]
        if (
            chunk.is_symlink()
            or not chunk.is_file()
            or chunk.stat().st_size != row["bytes"]
            or sha(chunk) != row["sha256"]
        ):
            raise ValueError("complete actual receipt-bound PV alias chunk set")
    if len(spec.get("alias_chunks", [])) != len(receipt.get("alias_chunks", [])):
        raise ValueError("all source alias chunks required")
    refs = {r["path"]: r["sha256"] for r in spec["alias_chunks"]}
    if any(refs.get(str((events.parent / r["file"]).resolve())) != r["sha256"]
           for r in receipt["alias_chunks"]):
        raise ValueError("all chunks exact source refs")
    audit = validate_audit(spec, reg, receipt)
    return reg, receipt, audit


def verify_h0_contract(contract, collection_reg):
    if (
        contract["phase"] != "human-prior-tdleaf-own-learning-v2"
        or contract["math"]["objective"]
        != "masked-MSE-tanh-authoritative-prior-logit-plus-residual-at-PV-leaf;TDLeaf-lambda0.5-v2"
        or contract["bootstrap_candidate_sha256"] != collection_reg["parent_candidate"]["sha256"]
        or contract["bootstrap_candidate_path"] != collection_reg["parent_candidate"]["path"]
        or contract["parent_admission_seal"] != collection_reg["parent_admission_seal"]
        or contract["parent_admission_result"] != collection_reg["parent_admission_result"]
        or contract["teacher_labels_used_in_own_phase"] is not False
        or contract["updates"] != 64
        or contract["tdleaf_lambda"] != 0.5
    ):
        raise ValueError("H0 current-parent weights-only TDLeaf contract; never teacher256")


def run(spec):
    if spec.get("schema") != "tdleaf-own-v2-root-native-chain-v1":
        raise ValueError("ROOT registered chain spec required")
    mode, seed = spec.get("mode"), spec.get("seed")
    if mode not in {"proof", "fresh-fit"} or seed not in {20262905, 20262906}:
        raise ValueError("fixed TDLeaf phase/seed only")
    reg, receipt, audit = validate_collection(spec)
    first, deadline = spec["first"], spec["deadline"]
    cap = 600 if mode == "proof" else 1800
    if not first <= time.time() < deadline <= min(
        first + cap, spec["operator_end_epoch"], ROOT_END
    ):
        raise ValueError("ROOT observed immutable phase clock")
    core = Path(reg["core_repo"])
    sys.path.insert(0, str(core / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if not first <= time.time() < deadline:
            raise TimeoutError("original phase clock; no reset")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace disk floor")

    guard()
    conversion_seal = read(spec["conversion_seal"])
    if (
        conversion_seal.get("schema") != "human-prior-tdleaf-dataset-conversion-seal-v2"
        or conversion_seal.get("registration") != spec["collection_registration"]
        or conversion_seal.get("receipt") != spec["collection_receipt"]
        or conversion_seal.get("events") != spec["events"]
        or conversion_seal.get("protected_aliases") != reg["protected_aliases"]
    ):
        raise ValueError("conversion seal must use exact actual collection refs")
    dataset, provenance = convert_sealed(conversion_seal)
    data_ref = spec["dataset"]
    provenance_ref = spec["target_provenance"]
    if (
        pinned(data_ref).read_bytes() != canonical(dataset) + b"\n"
        or pinned(provenance_ref).read_bytes() != canonical(provenance) + b"\n"
        or spec.get("conversion_result", {}).get("dataset") != data_ref
        or spec.get("conversion_result", {}).get("provenance") != provenance_ref
        or spec.get("conversion_result", {}).get("conversion_seal") != spec["conversion_seal"]
        or spec.get("conversion_result", {}).get("status")
        != "PASS-replayed-closed-PV-leaf-targets"
    ):
        raise ValueError("published conversion must byte-match a fresh full replay")
    guard()

    seal_ref = spec["phase_seal"]
    phase_seal = read(seal_ref)
    if (
        phase_seal.get("mode") != mode
        or phase_seal.get("seed") != seed
        or phase_seal.get("dataset") != data_ref
        or phase_seal.get("target_provenance") != provenance_ref
        or phase_seal.get("collection_audit_result") != spec["collection_audit_result"]
        or phase_seal.get("collection_audit_clock") != spec["collection_audit_clock"]
        or phase_seal.get("collection_audit_helper") != spec["collection_audit_helper"]
    ):
        raise ValueError("phase seal mode/seed differs")
    contract = contracts.build(phase_seal)
    verify_h0_contract(contract, reg)
    contract_ref = spec["contract"]
    if pinned(contract_ref).read_bytes() != canonical(contract) + b"\n":
        raise ValueError("phase contract must rederive exactly from its registered build seal")
    registration = read(spec["phase_registration"])
    if (
        registration.get("mode") != mode
        or registration.get("seed") != seed
        or registration.get("contract") != contract_ref
        or registration.get("contract_build_seal") != seal_ref
        or registration.get("dataset") != data_ref
        or registration.get("target_provenance") != provenance_ref
        or registration.get("output") != spec["phase_output"]
        or Path(contract_ref["path"]).resolve().is_relative_to(
            Path(registration["output"]).resolve()
        )
    ):
        raise ValueError("phase registration must reference contract outside output dir")
    if mode == "fresh-fit":
        proofs = spec.get("both_proof_results", {})
        if set(proofs) != {"20262905", "20262906"}:
            raise ValueError("both seed proofs required before either fresh-fit")
        for proof_seed, bundle in proofs.items():
            if set(bundle) != {"result", "contract", "collection_registration"}:
                raise ValueError("both proof bundles bind result, contract, and collection")
            proof_ref = bundle["result"]
            proof_contract = read(bundle["contract"])
            proof_collection = read(bundle["collection_registration"])
            proof = read(proof_ref)
            if (
                proof.get("status") != "PASS-TDLeaf-fixed-phase-and-fresh-native-loads-not-strength"
                or proof.get("mode") != "proof"
                or proof.get("contract_sha256") != bundle["contract"]["sha256"]
                or proof_contract.get("seed") != int(proof_seed)
                or proof_collection.get("seed") != int(proof_seed)
                or proof_contract.get("phase") != "human-prior-tdleaf-own-learning-v2"
                or proof_contract.get("bootstrap_candidate_sha256")
                != proof_collection.get("parent_candidate", {}).get("sha256")
                or proof.get("own_updates") != 8
                or proof.get("full_payload_bits_equal") is not True
            ):
                raise ValueError("both actual whole/pause/resume proof passes required")
        own_bundle = proofs[str(seed)]
        if (
            phase_seal.get("proof_result") != own_bundle["result"]
            or phase_seal.get("proof_contract") != own_bundle["contract"]
        ):
            raise ValueError("fresh-fit seal uses its seed's exact proof result and contract")
    guard()
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("prove.py")), "--registration",
         spec["phase_registration"]["path"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        start_new_session=True,
    )
    try:
        while proc.poll() is None:
            guard()
            time.sleep(0.1)
        output = proc.stdout.read() if proc.stdout else b""
        if proc.returncode:
            raise RuntimeError(f"proof/fresh-fit child failed: {output[-4000:]!r}")
    except BaseException:
        if proc.poll() is None:
            os.killpg(proc.pid, 15)
            proc.wait()
        raise
    result_path = Path(registration["output"]) / "result.json"
    result = json.loads(result_path.read_bytes())
    expected = "PASS-TDLeaf-fixed-phase-and-fresh-native-loads-not-strength"
    if result.get("status") != expected or result.get("mode") != mode:
        raise ValueError("inner proof receipt status/mode mismatch")
    return {
        "schema": "tdleaf-own-v2-root-native-chain-result-v1",
        "status": "PASS-phase-executed-not-strength",
        "seed": seed,
        "mode": mode,
        "audit": spec["collection_audit_result"],
        "dataset": data_ref,
        "provenance": provenance_ref,
        "contract": contract_ref,
        "phase_registration": spec["phase_registration"],
        "phase_result": {"path": str(result_path), "sha256": sha(result_path)},
        "teacher_labels_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(20262905, 20262906))
    parser.add_argument("--core", required=True, type=int, choices=(0, 2))
    parser.add_argument("--mode", required=True, choices=("proof", "fresh-fit"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 1800:
        raise ValueError("bounded wait only; original collection deadline remains binding")
    root = args.root.resolve()
    registration_path = root / "registration.json"
    if not registration_path.is_file():
        raise ValueError("original ROOT collection registration missing")
    collection_root = json.loads(registration_path.read_bytes())
    result_path = root / "result.json"
    wait_deadline = min(time.time() + args.wait_seconds, collection_root["deadline"])
    while not result_path.is_file() and time.time() < wait_deadline:
        time.sleep(min(1.0, max(0.0, wait_deadline - time.time())))
    spec = json.loads(args.spec.read_bytes())
    if (
        spec.get("seed") != args.seed
        or spec.get("mode") != args.mode
        or spec.get("root_collection_result", {}).get("path") != str(result_path)
        or spec.get("collection_registration", {}).get("path")
        != str((root / "collection-registration.json").resolve())
    ):
        raise ValueError("--root/--seed/--mode must match sealed actual collection inputs")
    phase_registration = read(spec["phase_registration"])
    if phase_registration.get("cpu_core") != args.core:
        raise ValueError("--core must match the pre-registered phase CPU")
    os.sched_setaffinity(0, {args.core})
    result = run(spec)
    output = Path(result["phase_result"]["path"]).parent / "chain-result.json"
    write_once({"path": str(output)}, result, 64 * 2**10)


if __name__ == "__main__":
    main()
