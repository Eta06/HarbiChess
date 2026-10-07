"""ROOT-controlled, phase-clocked integration for the closed TDLeaf v4 actors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

STAGE = Path(
    "/workspace/work/harbichess/continuation-20261007/"
    "tdleaf-human-prior-own-v2-native-integration-v1"
).resolve()
CHAIN_RUNNER = Path(__file__).resolve().with_name("run_root_chain.py")
ORIGINAL = Path(
    "/workspace/work/harbichess/continuation-20261007/"
    "tdleaf-human-prior-own-v2-qualified-producer-v3"
).resolve()
BASE = Path("/workspace/work/harbichess/continuation-20261007").resolve()
END_CAP = 1791448916.685839
SOURCE_INVENTORY_SHA = "750b2f64e4261704138d208a5dbd96a649d1599425ca3462d196c032a3648481"
SEEDS = (20262905, 20262906)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(262144), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("regular immutable file required")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("regular immutable file required")
    return {"path": str(path), "sha256": sha(path)}


def write_once(path, value, *, ram=False):
    path = Path(path).resolve()
    if ram and not path.is_relative_to("/dev/shm"):
        raise ValueError("large artifacts must stay in RAM")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return ref(path)


def load_pinned(r):
    p = Path(r["path"])
    if p.is_symlink():
        raise ValueError("pinned source changed")
    p = p.resolve(strict=True)
    if not p.is_file() or sha(p) != r["sha256"]:
        raise ValueError("pinned source changed")
    return json.loads(p.read_bytes())


def collect_refs(value):
    refs = {}

    def merge(children):
        for path, ref_value in children.items():
            if path in refs and refs[path] != ref_value:
                raise ValueError("conflicting references to one source path")
            refs[path] = ref_value

    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            raw_path = Path(value["path"])
            if raw_path.is_symlink():
                raise ValueError("source references cannot traverse symlinks")
            path = str(raw_path.resolve())
            normalized = {"path": path, "sha256": value["sha256"]}
            if path in refs and refs[path] != normalized:
                raise ValueError("conflicting references to one source path")
            refs[path] = normalized
        else:
            for child in value.values():
                merge(collect_refs(child))
    elif isinstance(value, list):
        for child in value:
            merge(collect_refs(child))
    return refs


def guard(first, deadline, budget):
    budget.check()
    if not first <= time.time() < deadline:
        raise TimeoutError("registered original phase clock expired; no reset")
    if shutil.disk_usage("/workspace").free < 256 * 2**20:
        raise RuntimeError("disk floor 256 MiB")


def run_child(command, output, first, deadline, budget, core):
    guard(first, deadline, budget)
    log = Path(output)
    with log.open("xb") as stream:
        proc = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        os.sched_setaffinity(proc.pid, {core})
        try:
            while proc.poll() is None:
                guard(first, deadline, budget)
                time.sleep(0.1)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise
        if proc.returncode:
            raise RuntimeError(f"registered child failed ({proc.returncode}); log={log}")
    guard(first, deadline, budget)
    return {"path": str(log), "sha256": sha(log), "finished": time.time()}


def root_sources(root, seed):
    root = Path(root).resolve(strict=True)
    if root.name != f"TDLeaf-collection-v4-actual-{seed}":
        raise ValueError("--root must be the exact actual v4 collection record directory")
    outer_path = root / "registration.json"
    outer = json.loads(outer_path.read_bytes())
    collection_path = root / "collection-registration.json"
    collection = json.loads(collection_path.read_bytes())
    source_inv = BASE / "tdleaf-human-prior-own-v2-qualified-producer-v3/source-inventory.json"
    if (
        outer.get("schema") != "ROOT-actual-TDLeaf-current-parent-collection-v3"
        or outer.get("seed") != seed
        or outer.get("source_inventory") != ref(source_inv)
        or sha(source_inv) != SOURCE_INVENTORY_SHA
        or collection.get("schema") != "human-prior-tdleaf-collection-registration-v2"
        or collection.get("seed") != seed
        or collection.get("status") != "registered"
    ):
        raise ValueError("actual v4 registration and exact original 750b source inventory required")
    return root, outer, collection, ref(outer_path), ref(collection_path), ref(source_inv)


def closed_collection(root, outer, collection, outer_ref, collection_ref, source_ref):
    result_path = root / "result.json"
    if not result_path.is_file():
        raise ValueError("collection is not closed; no receipt/result available")
    root_result_ref = ref(result_path)
    root_result = json.loads(result_path.read_bytes())
    if (
        root_result.get("status")
        != "PASS-actual1024-TDLeaf-current-parent-selfplay-not-strength"
        or root_result.get("collection_registration") != collection_ref
        or root_result.get("source_inventory") != source_ref
    ):
        raise ValueError("preserve and reject non-PASS or mismatched original collection")
    receipt_ref = root_result["raw_receipt"]
    receipt = load_pinned(receipt_ref)
    events_ref = ref(Path(collection["output_path"]) / "events.jsonl")
    alias_refs = []
    for row in receipt.get("alias_chunks", []):
        alias_refs.append(ref(Path(collection["output_path"]) / row["file"]))
    if (
        receipt.get("schema") != "human-prior-tdleaf-collection-receipt-v2"
        or receipt.get("status") != "PASS-exact-row-budget"
        or receipt.get("seed") != collection["seed"]
        or receipt.get("registration_sha256") != collection_ref["sha256"]
        or receipt.get("events_sha256") != events_ref["sha256"]
        or receipt.get("train_rows") != 1024
        or receipt.get("teacher_labels_used") is not False
        or receipt.get("parent_candidate_sha256") != collection["parent_candidate"]["sha256"]
        or receipt.get("producer_source_sha256") != collection["producer_source_sha256"]
    ):
        raise ValueError("exact closed original teacher-free 1024 actor receipt required")
    return root_result_ref, receipt_ref, events_ref, alias_refs


def begin_phase(record_dir, phase, operator_end):
    caps = {"convert": 600, "six": 600, "proof": 600, "fresh-fit": 1800}
    first = time.time()
    deadline = min(first + caps[phase], operator_end, END_CAP)
    if not first < deadline:
        raise TimeoutError("operator window expired")
    clock = {
        "schema": "ROOT-tdleaf-native-integration-phase-clock-v1",
        "phase": phase,
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": operator_end,
        "helper_sha256": sha(__file__),
    }
    clock_ref = write_once(record_dir / f"{phase}-clock.json", clock)
    return first, deadline, clock_ref


def load_runtime():
    sys.path.insert(0, str(STAGE))
    spec = importlib.util.spec_from_file_location(
        "tdleaf_integration_contracts", STAGE / "contracts.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


def phase_inputs(root, outer, collection, root_result_ref, collection_ref, source_ref,
                 receipt_ref, events_ref, alias_refs, record_dir, seed, phase, core):
    operator_end = outer["operator_end_epoch"]
    first, deadline, phase_clock_ref = begin_phase(record_dir, phase, operator_end)
    audit_dir = Path(f"/dev/shm/harbichess-tdleaf-native-integration-v1/{seed}")
    audit_dir.mkdir(parents=True, exist_ok=True)
    receipt = load_pinned(receipt_ref)
    audit_dir = audit_dir / phase
    return dict(
        first=first,
        deadline=deadline,
        clock=phase_clock_ref,
        audit_dir=audit_dir,
        operator_end=operator_end,
        receipt=receipt,
        root_result=root_result_ref,
        collection_ref=collection_ref,
        source_ref=source_ref,
        receipt_ref=receipt_ref,
        events_ref=events_ref,
        alias_refs=alias_refs,
    )


def execute(args):
    root, outer, collection, outer_ref, collection_ref, source_ref = root_sources(
        args.root, args.seed
    )
    if collection["cpu_core"] not in (0, 2) or args.core not in (0, 1, 2, 3):
        raise ValueError("actor core / execution CPU mismatch")
    os.sched_setaffinity(0, {args.core})
    sys.path.insert(0, str(Path(collection["core_repo"]) / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 2**30)
    if args.phase == "convert":
        root_result_ref, receipt_ref, events_ref, alias_refs = closed_collection(
            root, outer, collection, outer_ref, collection_ref, source_ref
        )
        meta_dir = BASE / f"TDLeaf-native-integration-v1-actual-{args.seed}"
        meta_dir.mkdir(exist_ok=True)
        phase = phase_inputs(root, outer, collection, root_result_ref, collection_ref,
                             source_ref, receipt_ref, events_ref, alias_refs,
                             meta_dir, args.seed, "convert", args.core)
        seal = {
            "schema": "human-prior-tdleaf-dataset-conversion-seal-v2",
            "registration": collection_ref,
            "receipt": receipt_ref,
            "events": events_ref,
            "alias_chunks": alias_refs,
            "protected_aliases": collection["protected_aliases"],
            "producer_directory": str(ORIGINAL),
            "features": {
                "path": str(Path(collection["parent_helpers"]["directory"]) / "model.py"),
                "sha256": collection["parent_helpers"]["model_sha256"],
            },
            "prior": {
                "path": collection["parent_helpers"]["prior_path"],
                "sha256": collection["parent_helpers"]["prior_sha256"],
            },
        }
        seal_ref = write_once(meta_dir / "conversion-seal.json", seal)
        conversion_dir = Path(
            f"/dev/shm/harbichess-human-prior-tdleaf-native-v1/{args.seed}/conversion"
        )
        log_ref = run_child(
            [sys.executable, str(STAGE / "convert_cli.py"), "--seal", seal_ref["path"],
             "--output", str(conversion_dir)],
            f"/dev/shm/harbichess-tdleaf-native-integration-v1/{args.seed}/convert.log",
            phase["first"], phase["deadline"], budget, args.core,
        )
        result_ref = ref(conversion_dir / "conversion-result.json")
        controller_result = {
            "schema": "tdleaf-native-integration-conversion-result-v1",
            "status": "PASS-converter-full-replay-not-strength",
            "seed": args.seed,
            "first": phase["first"],
            "deadline": phase["deadline"],
            "finished": time.time(),
            "source_inventory": source_ref,
            "collection_registration": collection_ref,
            "root_collection_result": root_result_ref,
            "receipt": receipt_ref,
            "events": events_ref,
            "audit_chunks": alias_refs,
            "conversion_seal": seal_ref,
            "conversion_result": result_ref,
            "command_log": log_ref,
        }
        return write_once(meta_dir / "conversion-controller-result.json", controller_result)

    if args.phase == "six":
        root_result_ref, receipt_ref, events_ref, alias_refs = closed_collection(
            root, outer, collection, outer_ref, collection_ref, source_ref
        )
        meta_dir = BASE / f"TDLeaf-native-integration-v1-actual-{args.seed}"
        meta_dir.mkdir(exist_ok=True)
        phase = phase_inputs(root, outer, collection, root_result_ref, collection_ref,
                             source_ref, receipt_ref, events_ref, alias_refs,
                             meta_dir, args.seed, "six", args.core)
        directory = phase["audit_dir"]
        directory.mkdir(parents=True, exist_ok=False)
        audit_clock = {
            "schema": "human-prior-tdleaf-six-root-audit-clock-v2",
            "first": phase["first"],
            "deadline": phase["deadline"],
            "cpu_core": args.core,
            "helper_sha256": sha(STAGE / "audit_collection_six.py"),
            "registration_sha256": collection_ref["sha256"],
            "receipt_sha256": receipt_ref["sha256"],
            "events_sha256": events_ref["sha256"],
            "producer_directory": str(ORIGINAL),
        }
        clock_ref = write_once(directory / "clock.json", audit_clock)
        result_path = Path(
            f"/dev/shm/harbichess-human-prior-tdleaf-native-v1/{args.seed}/six-audit.json"
        )
        command = [
            sys.executable,
            str(STAGE / "audit_collection_six.py"),
            "--registration", collection_ref["path"],
            "--receipt", receipt_ref["path"],
            "--events", events_ref["path"],
            "--clock", clock_ref["path"],
            "--output", str(result_path),
        ]
        log_ref = run_child(
            command,
            str(directory / "six.log"),
            phase["first"],
            phase["deadline"],
            budget,
            args.core,
        )
        audit_ref = ref(result_path)
        record = {
            "schema": "tdleaf-native-integration-six-audit-result-v1",
            "status": "PASS-six-actual-chronological-TDLeaf-PV-packets-and-full-alias-traces",
            "seed": args.seed,
            "collection_registration": collection_ref,
            "receipt": receipt_ref,
            "events": events_ref,
            "clock": clock_ref,
            "audit_result": audit_ref,
            "log": log_ref,
            "first": phase["first"],
            "deadline": phase["deadline"],
            "finished": time.time(),
        }
        return write_once(meta_dir / "six-controller-result.json", record)

    # The proof/fresh-fit phase specs are intentionally generated only from closed,
    # prior controller records. No phase clock is reused after failure.
    return run_phase(args, root, outer, collection, outer_ref, collection_ref, source_ref, budget)


def run_phase(args, root, outer, collection, outer_ref, collection_ref, source_ref, budget):
    seed = args.seed
    meta_dir = BASE / f"TDLeaf-native-integration-v1-actual-{seed}"
    root_result_ref, receipt_ref, events_ref, alias_refs = closed_collection(
        root, outer, collection, outer_ref, collection_ref, source_ref
    )
    conversion_record = json.loads((meta_dir / "conversion-controller-result.json").read_bytes())
    six_record = json.loads((meta_dir / "six-controller-result.json").read_bytes())
    for record, status in ((conversion_record, "PASS-converter-full-replay-not-strength"),
                           (six_record,
                            "PASS-six-actual-chronological-TDLeaf-PV-packets-and-full-alias-traces")):
        if record.get("status") != status or record.get("seed") != seed:
            raise ValueError("actual conversion and six-PV audit must pass before native phase")
    conversion_result_ref = conversion_record["conversion_result"]
    conversion_result = load_pinned(conversion_result_ref)
    dataset_ref, provenance_ref = conversion_result["dataset"], conversion_result["provenance"]
    phase = phase_inputs(root, outer, collection, root_result_ref, collection_ref,
                         source_ref, receipt_ref, events_ref, alias_refs,
                         meta_dir, seed, args.phase, args.core)
    phase_dir = phase["audit_dir"]
    phase_dir.mkdir(parents=True, exist_ok=False)
    seal = {
        "schema": "human-prior-tdleaf-contract-build-seal-v2",
        "status": "registered",
        "mode": args.phase,
        "first": phase["first"],
        "deadline": phase["deadline"],
        "operator_end_epoch": phase["operator_end"],
        "seed": seed,
        "parent_admission_seal": collection["parent_admission_seal"],
        "parent_admission_result": collection["parent_admission_result"],
        "collection_registration": collection_ref,
        "collection_receipt": receipt_ref,
        "target_provenance": provenance_ref,
        "dataset": dataset_ref,
        "collection_audit_result": six_record["audit_result"],
        "collection_audit_clock": six_record["clock"],
        "collection_audit_helper": ref(STAGE / "audit_collection_six.py"),
    }
    proof_result_ref = None
    proof_contract_ref = None
    if args.phase == "fresh-fit":
        proof_refs = {}
        for proof_seed in SEEDS:
            path = (
                BASE
                / f"TDLeaf-native-integration-v1-actual-{proof_seed}/proof-controller-result.json"
            )
            if not path.is_file():
                raise ValueError("both seed proof records are required before any fresh-fit")
            proof_record = json.loads(path.read_bytes())
            if proof_record.get("status") != "PASS-phase-executed-not-strength":
                raise ValueError("one or both ROOT proof phases did not pass")
            inner = load_pinned(proof_record["phase_result"])
            if (
                inner.get("status")
                != "PASS-TDLeaf-fixed-phase-and-fresh-native-loads-not-strength"
                or inner.get("mode") != "proof"
                or inner.get("own_updates") != 8
                or inner.get("full_payload_bits_equal") is not True
            ):
                raise ValueError("inner actual TDLeaf proof receipt did not pass")
            proof_refs[str(proof_seed)] = proof_record
        own = proof_refs[str(seed)]
        proof_result_ref = own["phase_result"]
        proof_contract_ref = own["contract"]
        seal["proof_result"] = proof_result_ref
        seal["proof_contract"] = proof_contract_ref
    seal_ref = write_once(phase_dir / "contract-build-seal.json", seal)
    contract_builder = load_runtime()
    contract = contract_builder.build(seal)
    contract_ref = write_once(
        Path(f"/dev/shm/harbichess-human-prior-tdleaf-native-v1/{seed}/{args.phase}-contract.json"),
        contract,
        ram=True,
    )
    registration = {
        "schema": "human-prior-tdleaf-training-orchestration-v2",
        "status": "registered",
        "mode": args.phase,
        "seed": seed,
        "cpu_core": args.core,
        "first": phase["first"],
        "deadline": phase["deadline"],
        "operator_end_epoch": phase["operator_end"],
        "contract_build_seal": seal_ref,
        "contract": contract_ref,
        "dataset": dataset_ref,
        "target_provenance": provenance_ref,
        "parent_candidate": collection["parent_candidate"],
        "train": ref(STAGE / "train.py"),
        "native": ref(STAGE / "native.py"),
        "inputs": collect_refs(contract["raw_collection_inputs"]),
        "source_sha256": contract["source_sha256"],
        "output": f"/dev/shm/harbichess-human-prior-tdleaf-native-v1/{seed}/{args.phase}",
    }
    registration_path = phase_dir / "training-registration.json"
    registration_ref = write_once(registration_path, registration)
    chain_spec = {
        "schema": "tdleaf-own-v2-root-native-chain-v1",
        "seed": seed,
        "mode": args.phase,
        "first": phase["first"],
        "deadline": phase["deadline"],
        "operator_end_epoch": phase["operator_end"],
        "root_collection_result": root_result_ref,
        "source_inventory": source_ref,
        "producer_directory": str(ORIGINAL),
        "collection_registration": collection_ref,
        "collection_receipt": receipt_ref,
        "events": events_ref,
        "alias_chunks": alias_refs,
        "root_collection_registration": outer_ref,
        "collection_audit_result": six_record["audit_result"],
        "collection_audit_clock": six_record["clock"],
        "collection_audit_helper": ref(STAGE / "audit_collection_six.py"),
        "conversion_seal": conversion_record["conversion_seal"],
        "conversion_result": conversion_result_ref,
        "dataset": dataset_ref,
        "target_provenance": provenance_ref,
        "phase_seal": seal_ref,
        "contract": contract_ref,
        "phase_registration": registration_ref,
        "phase_output": registration["output"],
    }
    if args.phase == "fresh-fit":
        chain_spec["both_proof_results"] = {
            str(proof_seed): {
                "result": proof_refs[str(proof_seed)]["phase_result"],
                "contract": proof_refs[str(proof_seed)]["contract"],
                "collection_registration": ref(
                    BASE / f"TDLeaf-collection-v4-actual-{proof_seed}/collection-registration.json"
                ),
            }
            for proof_seed in SEEDS
        }
    spec_ref = write_once(phase_dir / "root-chain-spec.json", chain_spec)
    stage = load_runtime()
    sys.modules.pop(stage.__name__, None)
    chain_path = CHAIN_RUNNER
    cmd = [sys.executable, str(chain_path), "--spec", spec_ref["path"]]
    # run_root_chain opens no second phase clock; it verifies the pre-registered one.
    log = run_child(
        cmd,
        phase_dir / "root-chain.log",
        phase["first"],
        phase["deadline"],
        budget,
        args.core,
    )
    chain_result = json.loads(
        (Path(registration["output"]) / "chain-result.json").read_bytes()
    )
    result = {
        "schema": "tdleaf-native-root-orchestration-phase-result-v1",
        "status": chain_result["status"],
        "seed": seed,
        "mode": args.phase,
        "phase_clock": phase["clock"],
        "phase_seal": seal_ref,
        "contract": contract_ref,
        "training_registration": registration_ref,
        "phase_result": chain_result["phase_result"],
        "chain_result": ref(Path(registration["output"]) / "chain-result.json"),
        "log": log,
        "first": phase["first"],
        "deadline": phase["deadline"],
        "finished": time.time(),
        "teacher_labels_used": False,
    }
    return write_once(meta_dir / f"{args.phase}-controller-result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--core", required=True, type=int, choices=(0, 1, 2, 3))
    parser.add_argument(
        "--phase", required=True, choices=("convert", "six", "proof", "fresh-fit")
    )
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
