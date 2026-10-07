"""ROOT-only strict fixed-own64/native/source/latency join BEFORE formal book selection."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from bindings import SEEDS, read, sha, validate_value_wiring


def collection_audits(known, audit_set):
    if (audit_set["schema"] != "NNUE-own-collection-replay-audit-set-v2"
            or len(audit_set["audits"]) != 2
            or {r["seed"] for r in audit_set["audits"]} != set(SEEDS)):
        raise ValueError("both actual six-packet audit receipts required")
    for row in audit_set["audits"]:
        seed = row["seed"]
        actual, clock = read(row["result"]), read(row["clock"])
        info = known["children"][str(seed)]
        receipt = read(info["collection_receipt"])
        if (actual["schema"] != "NNUE-own-collection-six-root-audit-result-v2"
                or actual["status"] !=
                "PASS-six-actual-chronological-parent-search-packets-and-full-alias-traces"
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
                receipt["periodic_independent_search_rows"]
                or len(actual["packets"]) != 6
                or any(actual[k] != 0
                       for k in ("new_games", "new_training_rows", "optimizer_updates"))):
            raise ValueError("exact12 actual replay inputs/packets/old clocks; no synthetic reuse")
    if sha(audit_set["helper"]["path"]) != audit_set["helper"]["sha256"]:
        raise ValueError("qualified replay helper bytes")


def check_known_metadata(spec):
    known = read(spec["known_protocol"])
    validate_value_wiring(known)
    profile, full, six = [read(spec[k]) for k in ("known48_profile", "known160_audit",
                                                "known_six_replay")]
    if (profile["status"] != "PASS-ownNNUE64-parent-trained-profile-not-strength"
            or profile["protocol_sha256"] != spec["known_protocol"]["sha256"]
            or len(profile["packets"]) != 48 or len(profile["parity"]) != 2
            or any(len(x["rows"]) != 24 or any(r["error"] > 1e-12 for r in x["rows"])
                   for x in profile["parity"])
            or len(profile["latency_ratios"]) != 2
            or any(x["learned_vs_parent_median"] > 1.10 or x["learned_vs_E8_median"] > 1.10
                   for x in profile["latency_ratios"])
            or profile["finished_epoch"] > profile["original_deadline_epoch"]):
        raise ValueError("actual48 parity/search/latency exact unchanged thresholds")
    if (full["status"] != "PASS-fullhistory-integrity" or full["games"] != 160
            or full["screen_result"] != "PASS-development-only"
            or full["protocol_sha256"] != spec["known_protocol"]["sha256"]
            or six["status"] != "PASS-six-actual-fixed-chronological-search-packets"
            or six["protocol_sha256"] != spec["known_protocol"]["sha256"]
            or six["profile_sha256"] != spec["known48_profile"]["sha256"]
            or len(six["packets"]) != 6 or six["finished"] > six["deadline"]):
        raise ValueError("complete BOTH-known160 point screens/integrity/six actual packets")
    collection_audits(known, read(spec["collection_audit_set"]))
    return known


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    spec = json.loads(a.spec.read_bytes())
    if not spec["first"] <= time.time() < spec["deadline"] <= min(
            spec["first"] + 900, spec["operator_end_epoch"]):
        raise ValueError("ROOT original eligibility900")
    known = check_known_metadata(spec)
    os.sched_setaffinity(0, {spec["cpu_core"]})
    sys.path.insert(0, known["core_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    memory = CgroupMemoryBudget(15 * 2**30)

    def guard():
        memory.check()
        if time.time() >= spec["deadline"]:
            raise TimeoutError("same original eligibility900 includes strict loads/reconciliation")
        if shutil.disk_usage("/workspace").free < 256 * 2**20:
            raise RuntimeError("workspace256MiB floor")

    from admission import child, import_nnue

    for name, digest in spec["helper_sha256"].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("eligibility helper closure source")
    import torch

    torch.set_num_threads(1)
    _, native, _, _ = import_nnue(known)
    admissions = []
    for seed in SEEDS:
        guard()
        admissions.append(child(known, seed, native)[2])
        guard()
    result = dict(schema="ONE-fixed-own64-candidate-eligibility-v2",
                  status="PASS-fixed-own64-bothseeds-eligible-no-formal-results", fixed_updates=64,
                  seeds=list(SEEDS), models=known["models"], formal_campaigns_used=0,
                  known_protocol=spec["known_protocol"], known48_profile=spec["known48_profile"],
                  known160_audit=spec["known160_audit"], known_six_replay=spec["known_six_replay"],
                  collection_audit_set=spec["collection_audit_set"], native_admissions=admissions,
                  source_and_helper_sha256=spec["helper_sha256"], specification_sha256=sha(a.spec),
                  first=spec["first"], deadline=spec["deadline"], finished_epoch=time.time(),
                  strength_qualified=False,
                  attribution=("teacher-once initializer; "
                               "own-search distillation gain still must confirm"))
    with a.output.open("x") as f:
        json.dump(result, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    main()
