"""Exact recorded collection-six audit joins; metadata only."""

from original_ownq_admission import SEEDS
from teacher_admission import read, sha


def collection_audits(known, audit_set):
    mc = known['target_variant'] == 'closed-terminal-mc-v1'
    set_schema = ('NNUE-own-closed-terminal-replay-audit-set-v1' if mc
                  else 'NNUE-own-collection-replay-audit-set-v2')
    result_schema = ('NNUE-own-closed-terminal-six-search-audit-v1' if mc
                     else 'NNUE-own-collection-six-root-audit-result-v2')
    result_status = ('PASS-six-actual-chronological-terminal-eligible-search-packets-and-complete-'
                     'alias-traces' if mc else
                     'PASS-six-actual-chronological-parent-search-packets-and-full-alias-traces')
    if (audit_set["schema"] != set_schema
            or len(audit_set["audits"]) != 2
            or {r["seed"] for r in audit_set["audits"]} != set(SEEDS)):
        raise ValueError("both actual six-packet audit receipts required")
    for row in audit_set["audits"]:
        seed = row["seed"]
        actual, clock = read(row["result"]), read(row["clock"])
        info = known["children"][str(seed)]
        receipt = read(info["collection_receipt"])
        if (actual["schema"] != result_schema
                or actual["status"] != result_status
                or actual["seed"] != seed
                or actual["helper_sha256"] != audit_set["helper"]["sha256"]
                or actual["registration_sha256"] != info["collection_registration"]["sha256"]
                or actual["receipt_sha256"] != info["collection_receipt"]["sha256"]
                or actual["events_sha256"] != info["events"]["sha256"]
                or actual["clock_sha256"] != row["clock"]["sha256"]
                or (actual["first"], actual["deadline"]) != (clock["first"], clock["deadline"])
                or not actual["first"] <= actual["finished"] <= actual["deadline"]
                <= actual["first"] + 600
                or [x["row_id"] for x in actual["packets"]] !=
                [receipt["training_row_ids"][i] for i in (0, 204, 409, 614, 819, 1023)]
                or len(actual["packets"]) != 6
                or any(actual[k] != 0
                       for k in ("new_games", "new_training_rows", "optimizer_updates"))):
            raise ValueError("exact12 actual replay inputs/packets/old clocks; no synthetic reuse")
    if sha(audit_set["helper"]["path"]) != audit_set["helper"]["sha256"]:
        raise ValueError("qualified replay helper bytes")

