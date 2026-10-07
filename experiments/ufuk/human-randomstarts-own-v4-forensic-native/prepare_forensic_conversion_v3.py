"""Prepare immutable v3 converter inputs after both six-packet audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forensic_audit_set import validate
from forensic_receipt import canonical, read, ref


def build(
    root: Path,
    seed: int,
    audit_set_path: Path,
    first: float,
    deadline: float,
    validator: Path,
    runtime: Path,
) -> dict:
    reg_path = root / f"human-random-collect-actual-{seed}" / "collection-registration.json"
    reg = read(reg_path)
    receipt_path = Path(reg["output_path"]) / "receipt.json"
    receipt = read(receipt_path)
    events = Path(reg["output_path"]) / "events.jsonl"
    view = Path(f"/dev/shm/harbichess-continuation-20261007/forensic-v3-{seed}/receipt-view.json")
    audit_set = json.loads(audit_set_path.read_bytes())
    validate(audit_set)
    item = next((x for x in audit_set["audits"] if x["seed"] == seed), None)
    refs = {
        key: ref(path)
        for key, path in (
            ("registration", reg_path),
            ("receipt", receipt_path),
            ("events", events),
            ("forensic_view", view),
        )
    }
    if item is None or any(item[k] != refs[k] for k in refs):
        raise ValueError("same-seed actual six-packet forensic audit required")
    helper = dict(path=str((root / "human-randomstarts-own-v2").resolve()))
    producer = Path(helper["path"])
    for name, digest in reg["producer_source_sha256"].items():
        if ref(producer / name)["sha256"] != digest:
            raise ValueError("original producer source closure")
    h = reg["parent_helpers"]
    return {
        "schema": "human-randomstarts-own1024-forensic-dataset-conversion-seal-v3",
        "status": "registered",
        "seed": seed,
        "first": first,
        "deadline": deadline,
        "operator_end_epoch": reg["operator_end_epoch"],
        **refs,
        "producer_directory": helper["path"],
        "procedural_bank_receipt": reg["procedural_bank_receipt"],
        "alias_chunks": {
            x["file"]: ref(Path(reg["output_path"]) / x["file"]) for x in receipt["alias_chunks"]
        },
        "features": ref(Path(h["directory"]) / "model.py"),
        "prior": ref(Path(h["prior_path"])),
        "core_repo": reg["core_repo"],
        "six_audit_set": ref(audit_set_path),
        "six_audit_validator": ref(validator),
        "six_audit_runtime": ref(runtime),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(20262905, 20262906), required=True)
    parser.add_argument("--audit-set", type=Path, required=True)
    parser.add_argument("--first", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal = build(
        args.root,
        args.seed,
        args.audit_set,
        args.first,
        args.deadline,
        args.validator,
        args.runtime,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(seal) + b"\n")
