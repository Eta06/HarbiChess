"""ROOT-registered producer only. This code does not train or call Stockfish."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from collector import AliasWriter, collect

HARD_MAX_SECONDS = 7200
OUTPUT_MAX = 128 * 2**20


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load(path, digest, name):
    path = Path(path).resolve()
    if sha(path) != digest:
        raise ValueError("pinned helper SHA differs: " + path.name)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if Path(mod.__file__).resolve() != path:
        raise ValueError("helper origin differs")
    return mod


def parent(reg):
    from parent_bridge import (
        admitted_candidate,
        require_collection_parent,
        validate_admission_result,
    )

    seal, admitted_contract = validate_admission_result(
        reg["parent_admission_seal"], reg["parent_admission_result"]
    )
    require_collection_parent(reg, seal, admitted_contract)

    h = reg["parent_helpers"]
    d = Path(h["directory"]).resolve()
    sys.path.insert(0, str(d))
    sys.modules.pop("model", None)
    sys.modules.pop("native", None)
    m = load(d / "model.py", h["model_sha256"], "model")
    n = load(d / "native.py", h["native_sha256"], "native")
    prior_mod = load(h["prior_path"], h["prior_sha256"], "ownq_prior")
    ext = load(h["extension_path"], h["extension_sha256"], "_kingbucket16")
    e = load(d / "evaluator.py", h["evaluator_sha256"], "ownq_evaluator")

    packet = admitted_candidate(reg["parent_admission_seal"], reg["parent_admission_result"], torch)
    c = packet["contract"]
    if (
        c["seed"] != reg["seed"]
        or reg["generation"] != c["generation"] + 1
        or c["feature_schema"] != m.FEATURE_SCHEMA
        or c["prior_helper_sha256"] != h["prior_sha256"]
        or hashlib.sha256(canon(c)).hexdigest() != reg["parent_candidate"]["contract_sha256"]
    ):
        raise ValueError("exact admitted current own-parent and unchanged model/prior")
    n.validate_weights(packet["model"])
    prior = e.AuthoritativePrior(prior_mod, prior_mod.ClassicalValue())
    return e.Evaluator(packet["model"], prior=prior, compiled=ext)


def execute(path):
    path = Path(path).resolve()
    reg = json.loads(Path(path).read_bytes())
    if (
        reg.get("schema") != "human-prior-own-collection-registration-v1"
        or reg.get("status") != "registered"
    ):
        raise ValueError("ROOT immutable registration required")
    if reg.get("seed") not in (20262905, 20262906) or reg.get("search") != {
        "nodes": 8192,
        "qdepth": 2,
        "max_depth": 8,
    }:
        raise ValueError("fixed seed/search contract")
    if (reg.get("row_limit"), reg.get("root_limit"), reg.get("plies_per_root")) != (
        1024,
        128,
        16,
    ):
        raise ValueError("fixed collection budget")
    for name in ("collector.py", "run_collection.py", "metadata_factory.py"):
        if sha(Path(__file__).with_name(name)) != reg["producer_source_sha256"][name]:
            raise ValueError("registered producer source differs")
    for path, digest in reg["generation_helper_sha256"].items():
        if sha(path) != digest:
            raise ValueError("generation helper changed")
    first, end, op_end = (
        reg[k]
        for k in (
            "original_first_epoch",
            "original_deadline_epoch",
            "operator_end_epoch",
        )
    )
    if (
        not first <= time.time() < end <= op_end
        or end - first > HARD_MAX_SECONDS
        or reg.get("output_path")
        != f"/dev/shm/harbichess-human-prior-ownq-v1/g-{reg['generation']}/{reg['seed']}"
    ):
        raise TimeoutError("ROOT observed phase/operator clock/output path")
    core = Path(reg["core_repo"]).resolve()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=core, text=True).strip() != reg[
        "core_commit"
    ] or subprocess.check_output(["git", "status", "--porcelain"], cwd=core, text=True):
        raise ValueError("clean core pin")
    os.sched_setaffinity(0, {reg["cpu_core"]})
    torch.set_num_threads(1)
    sys.path.insert(0, str(core / "src"))
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 2**30)
    root = Path(reg["root_pool"]["path"])
    protected_path = Path(reg["protected_aliases"]["path"])
    out = Path(reg["output_path"]).resolve()
    if not out.is_relative_to(Path("/dev/shm/harbichess-human-prior-ownq-v1")) or out.exists():
        raise ValueError("publish-once RAM output")
    if sha(root) != reg["root_pool"]["sha256"] or root.stat().st_size > 8 * 2**20:
        raise ValueError("4096-root pool pin/size")
    if (
        sha(protected_path) != reg["protected_aliases"]["sha256"]
        or protected_path.stat().st_size % 8
        or protected_path.stat().st_size > 256 * 2**20
    ):
        raise ValueError("protected projection input")
    import struct

    packed = protected_path.read_bytes()
    values = list(struct.unpack(f"<{len(packed) // 8}q", packed))
    if values != sorted(set(values)):
        raise ValueError("protected alias canonical encoding")

    def guard():
        budget.check()
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace disk floor")
        if time.time() >= end or time.time() >= op_end:
            raise TimeoutError("original collection/operator clock")

    guard()
    out.mkdir(parents=True, exist_ok=False)
    log = (out / "events.jsonl").open("xb")
    aliases = AliasWriter(out)

    def emit(event, check=True):
        if event["type"] == "search_row":
            event["row"]["search_alias_ref"] = aliases.append(event["row"].pop("eval_aliases"))
        log.write(canon(event) + b"\n")
        log.flush()
        os.fsync(log.fileno())
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > OUTPUT_MAX:
            raise RuntimeError("registered RAM artifact ceiling128MiB")
        if check:
            guard()

    try:
        search = load(reg["search_helper"]["path"], reg["search_helper"]["sha256"], "ownq_search")
        value = parent(reg)
        factory = SimpleNamespace(
            evaluator=value.nonterminal,
            make=lambda fn: search.BudgetSearch(
                fn, nodes=8192, quiescence_plies=2, max_depth=8, guard=guard
            ),
        )
        pool = json.loads(root.read_bytes())
        result = collect(
            pool,
            seed=reg["seed"],
            protected=set(values),
            factory=factory,
            guard=guard,
            on_event=emit,
        )
        aliases.close()
        audit_ids = [result["training_row_ids"][i] for i in (0, 204, 409, 614, 819, 1023)]
        chunks = [
            {"file": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(out.glob("search-aliases-*.bin"))
        ]
        log.flush()
        os.fsync(log.fileno())
        receipt = {
            "schema": "human-prior-own-collection-receipt-v1",
            "status": result["status"],
            "seed": reg["seed"],
            "registration_sha256": sha(path),
            "parent_candidate_sha256": reg["parent_candidate"]["sha256"],
            "parent_contract_sha256": reg["parent_candidate"]["contract_sha256"],
            "parent_helpers": reg["parent_helpers"],
            "parent_admission_result": reg["parent_admission_result"],
            "parent_admission_seal": reg["parent_admission_seal"],
            "generation": reg["generation"],
            "search_helper_sha256": reg["search_helper"]["sha256"],
            "root_pool_sha256": reg["root_pool"]["sha256"],
            "protected_aliases_sha256": reg["protected_aliases"]["sha256"],
            "ancestral_selection_path": reg["root_pool"]["selection_path"],
            "ancestral_selection_sha256": reg["root_pool"]["selection_sha256"],
            "ancestral_teacher_labels_sha256": reg["root_pool"]["ancestral_teacher_labels_sha256"],
            "source_selection_sha256": reg["root_pool"]["selection_sha256"],
            "source_ancestral_teacher_labels_sha256": reg["root_pool"][
                "ancestral_teacher_labels_sha256"
            ],
            "selected_root_order_sha256": result["selected_root_order_sha256"],
            "train_rows": len(result["training_row_ids"]),
            "all_actor_rows": len(result["rows"]),
            "starts_considered": result["starts_considered"],
            "training_row_ids": result["training_row_ids"],
            "periodic_independent_search_rows": audit_ids,
            "games": result["games"],
            "unique_exposure_aliases": len(result["exposed_piece_aliases"]),
            "alias_chunks": chunks,
            "events_bytes": (out / "events.jsonl").stat().st_size,
            "events_sha256": sha(out / "events.jsonl"),
            "search": reg["search"],
            "teacher_labels_used": False,
            "target_source": result["target_source"],
            "original_first_epoch": first,
            "original_deadline_epoch": end,
            "operator_end_epoch": op_end,
            "finished_epoch": time.time(),
            "producer_source_sha256": reg["producer_source_sha256"],
            "generation_helper_sha256": reg["generation_helper_sha256"],
            "exposed_board_alias_count": len(result["exposed_piece_aliases"]),
            "exposure_definition": (
                "all path boards plus actual static evaluator inputs; "
                "per-row evaluator aliases in sorted-unique sidecars"
            ),
        }
        guard()
        with (out / "receipt.json").open("xb") as f:
            f.write(canon(receipt) + b"\n")
            f.flush()
            os.fsync(f.fileno())
    except BaseException as exc:
        aliases.close()
        emit({"type": "producer-failure", "error": repr(exc), "time": time.time()}, False)
        raise
    finally:
        log.flush()
        os.fsync(log.fileno())
        log.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registration", required=True, type=Path)
    execute(p.parse_args().registration)
