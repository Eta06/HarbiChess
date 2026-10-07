"""Strict pair of six-packet forensic audits; no games or optimization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "human-randomstarts-own-forensic-replay-audit-set-v3"
RESULT_SCHEMA = "human-randomstarts-own-six-root-audit-result-forensic-v3"
CLOCK_SCHEMA = "human-randomstarts-own-six-root-audit-clock-forensic-v3"
SEEDS = (20262905, 20262906)
OFFSETS = (0, 204, 409, 614, 819, 1023)
STATUS = "PASS-six-actual-chronological-parent-search-packets-and-full-alias-traces"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(262144), b""):
            h.update(block)
    return h.hexdigest()


def read(ref: dict) -> dict:
    path = Path(ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("forensic audit artifact changed")
    return json.loads(path.read_bytes())


def validate(manifest: dict) -> dict:
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("status") != "PASS-both-forensic-six-packet-audits"
    ):
        raise ValueError("forensic audit-set schema/status")
    audits = manifest.get("audits")
    helper = manifest.get("helper")
    validator = manifest.get("validator")
    if (
        not isinstance(helper, dict)
        or not isinstance(validator, dict)
        or sha(helper["path"]) != helper["sha256"]
        or sha(validator["path"]) != validator["sha256"]
    ):
        raise ValueError("forensic audit-set validator/helper pins")
    if not isinstance(audits, list) or tuple(sorted(x.get("seed") for x in audits)) != SEEDS:
        raise ValueError("both fixed seed audits required exactly once")
    verified = []
    for item in sorted(audits, key=lambda x: x["seed"]):
        seed = item["seed"]
        registration, receipt, view = (
            read(item[key]) for key in ("registration", "receipt", "forensic_view")
        )
        if sha(item["events"]["path"]) != item["events"]["sha256"]:
            raise ValueError("closed event bytes changed")
        result, clock = read(item["result"]), read(item["clock"])
        packets = result.get("packets")
        if (
            registration.get("seed") != seed
            or registration.get("schema") != "human-randomstarts-own-collection-registration-v2"
            or registration.get("status") != "registered"
            or receipt.get("seed") != seed
            or receipt.get("schema") != "human-randomstarts-own-collection-receipt-v2"
            or receipt.get("status") != "PASS-exact-row-budget"
            or receipt.get("train_rows") != 1024
            or receipt.get("teacher_labels_used") is not False
            or receipt.get("generation") != registration.get("generation")
            or receipt.get("registration_sha256") != item["raw_wrong_registration_sha256"]
            or receipt.get("registration_sha256") == item["registration"]["sha256"]
            or receipt.get("events_sha256") != item["events"]["sha256"]
            or receipt.get("training_row_ids") != item["training_row_ids"]
            or receipt.get("periodic_independent_search_rows") != item["periodic_search_rows"]
            or view.get("schema") != "human-randomstarts-own-forensic-receipt-view-v3"
            or view.get("status") != "PASS-source-shadow-only-audit-required"
            or view.get("seed") != seed
            or view.get("generation") != registration.get("generation")
            or view.get("raw_receipt") != item["receipt"]
            or view.get("original_registration") != item["registration"]
            or view.get("raw_receipt_registration_sha256") != item["raw_wrong_registration_sha256"]
            or view.get("verified_original_registration_sha256") != item["registration"]["sha256"]
            or result.get("schema") != RESULT_SCHEMA
            or result.get("status") != STATUS
            or result.get("seed") != seed
            or result.get("generation") != registration.get("generation")
            or result.get("parent_admission_result") != registration.get("parent_admission_result")
            or result.get("helper_sha256") != manifest.get("helper", {}).get("sha256")
            or clock.get("schema") != CLOCK_SCHEMA
            or clock.get("seed") != seed
            or clock.get("helper_sha256") != result.get("helper_sha256")
            or clock.get("operator_end_epoch") != item["operator_end_epoch"]
            or result.get("clock_sha256") != item["clock"]["sha256"]
            or result.get("registration_sha256") != item["registration"]["sha256"]
            or result.get("receipt_sha256") != item["receipt"]["sha256"]
            or result.get("events_sha256") != item["events"]["sha256"]
            or result.get("forensic_view_sha256") != item["forensic_view"]["sha256"]
            or result.get("raw_receipt_registration_sha256")
            != item["raw_wrong_registration_sha256"]
            or result.get("verified_original_registration_sha256") != item["registration"]["sha256"]
            or result.get("receipt_view_is_not_a_rewritten_v2_receipt") is not True
            or not isinstance(packets, list)
            or len(packets) != 6
            or any(
                result.get(k) != 0 for k in ("new_training_rows", "new_games", "optimizer_updates")
            )
            or (result.get("first"), result.get("deadline"))
            != (clock.get("first"), clock.get("deadline"))
            or not result.get("first", 0) < result.get("finished", 0) <= result.get("deadline", 0)
            or result.get("deadline", 0)
            > min(result.get("first", 0) + 600, item["operator_end_epoch"])
        ):
            raise ValueError("forensic six-audit result/clock/input binding")
        if [p.get("row_id") for p in packets] != item.get("periodic_search_rows"):
            raise ValueError("six exact chronological packet rows")
        expected = [item["training_row_ids"][i] for i in OFFSETS]
        if expected != item["periodic_search_rows"]:
            raise ValueError("raw source receipt's exact fixed packet ordinals")
        if (
            clock.get("registration_sha256") != item["registration"]["sha256"]
            or clock.get("receipt_sha256") != item["receipt"]["sha256"]
            or clock.get("events_sha256") != item["events"]["sha256"]
            or clock.get("forensic_view_sha256") != item["forensic_view"]["sha256"]
            or clock.get("registration") != item["registration"]
            or clock.get("receipt") != item["receipt"]
            or clock.get("events") != item["events"]
            or clock.get("forensic_view") != item["forensic_view"]
            or result.get("raw_receipt_path") != item["receipt"]["path"]
            or result.get("forensic_view_path") != item["forensic_view"]["path"]
        ):
            raise ValueError("audit clock exact forensic references")
        verified.append({"seed": seed, "result": item["result"], "clock": item["clock"]})
    return {"schema": SCHEMA, "status": "PASS-both-forensic-six-packet-audits", "audits": verified}
