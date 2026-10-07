"""ROOT-only final statistical AND source/native/latency/ONE lineage qualification join."""

import argparse
import json
import time
from pathlib import Path

from bindings import SEEDS, read, sha
from eligibility import check_known_metadata
from select_books import publish


def validate_bootstrap(bootstrap):
    expected_gates = {
        "E8_direct_gt_060", "E8_direct_lower_gt_050", "SF_gain_E8_gt_010",
        "SF_gain_E8_lower_gt_0", "parent_direct_gt_060", "parent_direct_lower_gt_050",
        "SF_gain_parent_gt_0", "SF_gain_parent_lower_gt_0", "final_SF_ge_025", "caps_le_005",
    }
    if set(map(str, bootstrap)) != set(map(str, SEEDS)):
        raise ValueError("exact BOTH seed bootstrap reports; no vacuous pass")
    passed = True
    for seed in SEEDS:
        row = bootstrap.get(str(seed), bootstrap.get(seed))
        if (row["seed"] != seed or row["root_blocks"] != 48 or row["games"] != 480
                or row["bootstrap_replicates"] != 50000
                or row["confidence_two_sided"] != .9984375
                or row["lower_tail"] != .00078125
                or len(row["analyses"]) != 2):
            raise ValueError("frozen bootstrap count/allocation/paired-root blocks")
        if [a["adverse_caps"] for a in row["analyses"]] != [False, True]:
            raise ValueError("raw AND adverse caps analysis")
        for analysis in row["analyses"]:
            if (len(analysis["intervals"]) != 4 or set(analysis["gates"]) != expected_gates
                    or any(type(v) is not bool for v in analysis["gates"].values())):
                raise ValueError("all original intervals and numerical gates retained")
            actual = all(analysis["gates"].values())
            if analysis["passed"] != actual:
                raise ValueError("bootstrap gate conjunction mismatch")
            passed &= actual
    return bool(passed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.spec.read_bytes())
    if not spec["first"] <= time.time() < spec["deadline"] <= min(
            spec["first"] + 900, spec["operator_end_epoch"]):
        raise ValueError("ROOT fresh observed finaljoin clock, not strength rerun/reset")
    registration = read(spec["formal_registration"])
    known_for_guard = read(registration["known_protocol"])
    import os
    import shutil
    import sys

    os.sched_setaffinity(0, {spec["cpu_core"]})
    sys.path.insert(0, known_for_guard["core_repo"] + "/src")
    from harbichess.training.cgroup_budget import CgroupMemoryBudget

    budget = CgroupMemoryBudget(15 * 2**30)
    budget.check()
    if shutil.disk_usage("/workspace").free < 256 * 2**20:
        raise RuntimeError("workspace256MiB floor")
    audit = read(spec["formal_audit"])
    eligibility = read(registration["eligibility"])
    if (audit["schema"] != "ONE-formal960-independent-audit-v2"
            or audit["registration_sha256"] != spec["formal_registration"]["sha256"]
            or audit["integrity"] != "PASS-all960-histories" or audit["total_games"] != 960
            or audit["finished"] > audit["deadline"]
            or eligibility["status"] != "PASS-fixed-own64-bothseeds-eligible-no-formal-results"
            or eligibility["seeds"] != list(SEEDS) or eligibility["fixed_updates"] != 64
            or eligibility["models"] != read(registration["known_protocol"])["models"]):
        raise ValueError("all actual960/integrity/original-clock/exact fixed endpoint lineage")
    known = check_known_metadata(registration)
    # FULL strict native/source/optimizer/RNG proof checks again; no new training or search.
    import torch
    from admission import child, import_nnue

    torch.set_num_threads(1)
    _, native, _, _ = import_nnue(known)
    admissions = [child(known, seed, native)[2] for seed in SEEDS]
    binding = read(spec["supplement_final_binding"])
    if (binding["registration_sha256"] != spec["formal_registration"]["sha256"]
            or binding["supplement"] != registration["supplement"]
            or binding["parameter_changes"] is not False):
        raise ValueError("registered supplement unchanged before actualformalresults")
    budget.check()
    if len(audit["betting"]["records"]) != 8 or audit["betting"]["alpha_each"] != 0.00078125:
        raise ValueError("all8 fixed exactbound alpha unchanged")
    if any(sha(Path(__file__).with_name(name)) != digest
           for name, digest in registration["helper_sha256"].items()):
        raise ValueError("frozen formal helper closure changed")
    expected_metrics = {"E8_direct", "SF_gain_E8", "parent_direct", "SF_gain_parent"}
    records = audit["betting"]["records"]
    if {(r["seed"], r["metric"]) for r in records} != {
            (s, m) for s in SEEDS for m in expected_metrics}:
        raise ValueError("exactbothseeds/all8 required betting LCBs")
    betting_pass = all(r["lower"] > (0.5 if r["metric"].endswith("direct") else 0.)
                       for r in records)
    bootstrap_pass = validate_bootstrap(audit["bootstrap"])
    if audit["statistical_gates_pass"] != (betting_pass and bootstrap_pass):
        raise ValueError("recomputed conjunction; cannot substitute success boolean")
    verified_books = read(registration["book_verification"])
    if (verified_books["status"] !=
            "PASS96-fullhistory-draw-source-probability-and-current-DAG-exclusion"
            or verified_books["books"] != registration["books"]):
        raise ValueError("independent source draw/exclusion proof")
    if time.time() >= spec["deadline"]:
        raise TimeoutError("same finaljoin900 includes strict native/source checks")
    result = dict(schema="ONE-ancestry-conditional-ownNNUE64-final-qualification-v2",
                  status="PASS-conditional-independent-confirmation"
                  if audit["statistical_gates_pass"] else "FAIL-fixed-confirmation-gates",
                  strength_qualified=bool(audit["statistical_gates_pass"]),
                  native_admissions=admissions,
                  formal_registration=spec["formal_registration"],
                  formal_audit=spec["formal_audit"],
                  exact_fixed_updates=64, seeds=list(SEEDS), games=960,
                  old_MAX8="closed", new_joint_error_bound=0.00625,
                  cumulative_old_plus_new_bound=0.05625, global_error_005_claim=False,
                  attribution="own-search distillation gain beyond exact teacher-once parent",
                  scope=("conditional independently randomized source/ECO strata; "
                         "not global virgin/Elo"),
                  no_novelty_claim=True, no_fallback_or_retest=True,
                  first=spec["first"], deadline=spec["deadline"], finished=time.time(),
                  helper_sha256=sha(__file__))
    publish(a.output, result)


if __name__ == "__main__":
    main()
