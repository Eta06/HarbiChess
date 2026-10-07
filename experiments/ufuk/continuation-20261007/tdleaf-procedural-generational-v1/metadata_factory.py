"""Pure versioned current-own-parent collection registration; no model calls."""

import argparse
import json
import time
from pathlib import Path

from parent_bridge import (
    canonical,
    pinned,
    require_collection_parent,
    sha,
    validate_admission_result,
)

HERE = Path(__file__).resolve().parent
END = 1791448916.685839
SEARCH_SHAS = {
    "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670",
}


def ref(p):
    p = Path(p).resolve()
    return {"path": str(p), "sha256": sha(p)}


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(canonical(value) + b"\n")


def collection(spec, output):
    if (
        spec["schema"] != "procedural-generational-tdleaf-collection-build-seal-v2"
        or spec["status"] != "registered"
    ):
        raise ValueError("ROOT registered human-prior source/clock seal")
    first, end, op = spec["first"], spec["deadline"], spec["operator_end_epoch"]
    if not first <= time.time() < end <= min(first + 10800, op, END):
        raise ValueError("new original collection clock")
    parent, contract = validate_admission_result(
        spec["parent_admission_seal"], spec["parent_admission_result"]
    )
    generation, seed = parent["generation"], parent["seed"]
    from generation_plan import validate_bank_slot, validate_plan
    from root_bank import verify

    plan = validate_plan(spec["generation_plan"], seed)
    if first < plan["first"] or op != plan["operator_end_epoch"]:
        raise ValueError("original generation clock inside preregistered plan")
    if not 1 <= generation <= 3:
        raise ValueError("only fixed three generations")
    if generation > 1 and contract["generation_plan"] != spec["generation_plan"]:
        raise ValueError("fixed same plan across all three generations")
    bank_ref = plan["banks"][str(seed)][str(generation)]
    if spec["procedural_bank_receipt"] != bank_ref:
        raise ValueError("precommitted generation bank")

    def guard():
        if time.time() >= end:
            raise TimeoutError("source replay charged to original collection clock")

    selected, bank = verify(bank_ref, spec["protected_aliases"], guard)
    _, previous_aliases = validate_bank_slot(plan, seed, generation, bank, selected)
    selection_ref = bank["selection"]
    pool = dict(
        schema="generational-procedural-train-roots-v1",
        selection_status="pass",
        train_only=True,
        source_selection_sha256=selection_ref["sha256"],
        procedural_receipt_sha256=bank_ref["sha256"],
        previous_start_aliases=previous_aliases,
        rows=selected["rows"],
    )
    # No teacher label path or content is consumed by this factory.
    pool_path = Path(spec["root_pool_output"])
    publish(pool_path, pool)
    protected = pinned(spec["protected_aliases"])
    if protected.stat().st_size % 8:
        raise ValueError("int64 protection")
    h = spec["parent_helpers"]
    if (
        h["model_sha256"] != sha(HERE / "model.py")
        or h["prior_sha256"] != contract["prior_helper_sha256"]
    ):
        raise ValueError("unchanged architecture/prior")
    for path, key in [
        (Path(h["directory"]) / "model.py", "model_sha256"),
        (Path(h["directory"]) / "native.py", "native_sha256"),
        (Path(h["directory"]) / "evaluator.py", "evaluator_sha256"),
        (Path(h["prior_path"]), "prior_sha256"),
        (Path(h["extension_path"]), "extension_sha256"),
    ]:
        pinned({"path": str(path), "sha256": h[key]})
    pinned(spec["search_helper"])
    if (
        spec["search_helper"]["sha256"] not in SEARCH_SHAS
        or spec["search_helper"] != contract["search_helper"]
    ):
        raise ValueError("original de53 search")
    pv_search = ref(HERE / "search_pv.py")
    reg = dict(
        schema="procedural-generational-tdleaf-collection-registration-v2",
        protection_scope_schema="procedural-generational-tdleaf-current-root-query-path-v2",
        status="registered",
        generation=generation,
        seed=seed,
        cpu_core=spec["cpu_core"],
        original_first_epoch=first,
        original_deadline_epoch=end,
        operator_end_epoch=op,
        core_repo=contract["core_source_repo"],
        core_commit=contract["core_source_commit"],
        output_path=f"/dev/shm/harbichess-procedural-generational-tdleaf-v2/g-{generation}/{seed}",
        parent_candidate={
            **parent["parent_candidate"],
            "contract_sha256": sha_canonical(contract),
        },
        parent_admission_seal=spec["parent_admission_seal"],
        parent_admission_result=spec["parent_admission_result"],
        parent_helpers=h,
        root_pool=ref(pool_path),
        generation_plan=spec["generation_plan"],
        procedural_bank_receipt=bank_ref,
        protected_aliases=spec["protected_aliases"],
        search_helper=spec["search_helper"],
        pv_search_helper=pv_search,
        search=dict(nodes=8192, qdepth=2, max_depth=8),
        row_limit=2048,
        root_limit=512,
        plies_per_root=16,
        producer_source_sha256={
            n: sha(HERE / n)
            for n in (
                "collector.py",
                "run_collection.py",
                "metadata_factory.py",
                "capture_record.py",
                "tdleaf_targets.py",
                "search_pv.py",
            )
        },
        generation_helper_sha256={
            str(HERE / n): sha(HERE / n)
            for n in ("parent_bridge.py", "generation_plan.py", "root_bank.py")
        },
        teacher_labels_used=False,
    )
    require_collection_parent(reg, parent, contract)
    publish(output, reg)
    return reg


def sha_canonical(x):
    import hashlib

    return hashlib.sha256(canonical(x)).hexdigest()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    collection(json.loads(a.seal.read_bytes()), a.output)
