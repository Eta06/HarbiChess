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
    "90c85ba8e3401cba57c82e6c2a34764d25aa999a03b244a4f2838124a8b46357",
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
        spec["schema"] != "human-prior-tdleaf-collection-build-seal-v2"
        or spec["status"] != "registered"
    ):
        raise ValueError("ROOT registered human-prior source/clock seal")
    first, end, op = spec["first"], spec["deadline"], spec["operator_end_epoch"]
    if not first <= time.time() < end <= min(first + 7200, op, END):
        raise ValueError("new original collection clock")
    parent, contract = validate_admission_result(
        spec["parent_admission_seal"], spec["parent_admission_result"]
    )
    generation, seed = parent["generation"], parent["seed"]
    source = json.loads(pinned(spec["ancestral_root_pool"]).read_bytes())
    selection_ref = spec["ancestral_selection"]
    selected = json.loads(pinned(selection_ref).read_bytes())
    if (
        source["schema"] != "teacher-selected-ownq-train-roots-v2"
        or source["selection_status"] != "pass"
        or source["train_only"] is not True
        or source["source_selection_sha256"] != selection_ref["sha256"]
        or len(source["rows"]) != 4096
        or len(selected["rows"]) != 4096
    ):
        raise ValueError("fixed original legal TRAIN-only4096 projection ancestry")
    for old, new in zip(selected["rows"], source["rows"], strict=True):
        expected = dict(
            root_id=old["row_id"],
            source_row_id=old["row_id"],
            trajectory_id=old["trajectory_id"],
            root_fen=old["root_fen"],
            prefix_uci=old["prefix_uci"],
            role="TRAIN",
        )
        if any(new.get(k) != v for k, v in expected.items()):
            raise ValueError("exact ancestral TRAIN projection")
    pool = dict(
        schema="human-prior-own-train-roots-v2",
        selection_status="pass",
        train_only=True,
        source_selection_sha256=selection_ref["sha256"],
        ancestral_teacher_labels_sha256=source["source_teacher_labels_sha256"],
        rows=source["rows"],
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
        schema="human-prior-tdleaf-collection-registration-v2",
        protection_scope_schema="human-prior-tdleaf-current-root-query-path-v2",
        status="registered",
        generation=generation,
        seed=seed,
        cpu_core=spec["cpu_core"],
        original_first_epoch=first,
        original_deadline_epoch=end,
        operator_end_epoch=op,
        core_repo=contract["core_source_repo"],
        core_commit=contract["core_source_commit"],
        output_path=f"/dev/shm/harbichess-human-prior-tdleaf-v2/g-{generation}/{seed}",
        parent_candidate={
            **parent["parent_candidate"],
            "contract_sha256": sha_canonical(contract),
        },
        parent_admission_seal=spec["parent_admission_seal"],
        parent_admission_result=spec["parent_admission_result"],
        parent_helpers=h,
        root_pool=dict(
            **ref(pool_path),
            selection_path=selection_ref["path"],
            selection_sha256=selection_ref["sha256"],
            ancestral_teacher_labels_sha256=pool["ancestral_teacher_labels_sha256"],
        ),
        protected_aliases=spec["protected_aliases"],
        search_helper=spec["search_helper"],
        pv_search_helper=pv_search,
        search=dict(nodes=8192, qdepth=2, max_depth=8),
        row_limit=1024,
        root_limit=128,
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
        generation_helper_sha256={str(HERE / n): sha(HERE / n) for n in ("parent_bridge.py",)},
        ancestral_pool=spec["ancestral_root_pool"],
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
