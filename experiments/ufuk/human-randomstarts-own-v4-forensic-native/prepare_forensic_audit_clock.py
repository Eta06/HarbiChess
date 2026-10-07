"""Prepare one ROOT-observed forensic six-packet replay clock."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from forensic_receipt import canonical, derive_view, read, ref

SCHEMA = "human-randomstarts-own-six-root-audit-clock-forensic-v3"


def build(
    root: Path, seed: int, first: float, deadline: float, cpu_core: int, helper: Path
) -> dict:
    if seed not in (20262905, 20262906) or not first <= time.time() < deadline <= first + 600:
        raise ValueError("ROOT-observed original600 audit clock")
    registration = root / f"human-random-collect-actual-{seed}" / "collection-registration.json"
    reg = read(registration)
    receipt = Path(reg["output_path"]) / "receipt.json"
    events = Path(reg["output_path"]) / "events.jsonl"
    view = Path(f"/dev/shm/harbichess-continuation-20261007/forensic-v3-{seed}/receipt-view.json")
    if not view.is_file() or json.loads(view.read_bytes()) != derive_view(
        registration, receipt, root / "human-randomstarts-own-v2"
    ):
        raise ValueError("ROOT-published matching immutable forensic view required")
    if deadline > reg["operator_end_epoch"]:
        raise ValueError("audit within original operator end")
    registration_ref, receipt_ref, events_ref, view_ref = map(
        ref, (registration, receipt, events, view)
    )
    return {
        "schema": SCHEMA,
        "seed": seed,
        "first": first,
        "deadline": deadline,
        "cpu_core": cpu_core,
        "helper_sha256": ref(helper)["sha256"],
        "registration": registration_ref,
        "registration_sha256": registration_ref["sha256"],
        "receipt": receipt_ref,
        "receipt_sha256": receipt_ref["sha256"],
        "events": events_ref,
        "events_sha256": events_ref["sha256"],
        "forensic_view": view_ref,
        "forensic_view_sha256": view_ref["sha256"],
        "producer_directory": str((root / "human-randomstarts-own-v2").resolve()),
        "operator_end_epoch": reg["operator_end_epoch"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--first", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--cpu-core", type=int, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.root, args.seed, args.first, args.deadline, args.cpu_core, args.helper)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
