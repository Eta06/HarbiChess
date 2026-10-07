"""Build a ROOT-sealed forensic replay audit set from completed v3 audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
        / "human-randomstarts-own-v3-forensic-shadow"
    ),
)

from forensic_audit_set import OFFSETS, SCHEMA, validate
from forensic_receipt import derive_view, read
from forensic_receipt import ref as make_ref


def build(root: Path, audit_helper_path: Path, validator_path: Path) -> dict:
    audits = []
    for seed in (20262905, 20262906):
        registration = (
            root
            / f"human-random-collect-actual-{seed}"
            / "collection-registration.json"
        )
        reg = read(registration)
        receipt = Path(reg["output_path"]) / "receipt.json"
        events = Path(reg["output_path"]) / "events.jsonl"
        producer = root / "human-randomstarts-own-v2"
        view = Path(
            f"/dev/shm/harbichess-continuation-20261007/forensic-v3-{seed}/receipt-view.json"
        )
        expected_view = derive_view(registration, receipt, producer)
        if not view.is_file() or json.loads(view.read_bytes()) != expected_view:
            raise ValueError(
                "ROOT-published exact derived forensic receipt view required"
            )
        result_dir = Path(
            f"/dev/shm/harbichess-human-randomstarts-forensic-audit-v3/{seed}"
        )
        result_path, clock_path = result_dir / "result.json", result_dir / "clock.json"
        raw = read(receipt)
        rref, qref, eref, vref = map(make_ref, (registration, receipt, events, view))
        audits.append(
            {
                "seed": seed,
                "registration": rref,
                "receipt": qref,
                "events": eref,
                "forensic_view": vref,
                "result": make_ref(result_path),
                "clock": make_ref(clock_path),
                "raw_wrong_registration_sha256": raw["registration_sha256"],
                "operator_end_epoch": reg["operator_end_epoch"],
                "training_row_ids": raw["training_row_ids"],
                "periodic_search_rows": [raw["training_row_ids"][i] for i in OFFSETS],
            }
        )
    manifest = {
        "schema": SCHEMA,
        "status": "PASS-both-forensic-six-packet-audits",
        "helper": make_ref(audit_helper_path),
        "validator": make_ref(validator_path),
        "audits": audits,
    }
    validate(manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-helper", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = (
        json.dumps(
            build(args.root, args.audit_helper, args.validator),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
