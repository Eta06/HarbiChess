"""ROOT-only formal metadata seal AFTER fixed eligibility; no selection/inference/games."""

import argparse
import json
import os
import time
from pathlib import Path

from bindings import SEEDS, protocol_views, read, sha, validate_supplement


def publish(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def formal_duration(spec):
    return spec["deadline"] - spec["first"]


def seal(spec, directory):
    eligibility = read(spec["eligibility"])
    if (eligibility["schema"] != "ONE-fixed-own64-candidate-eligibility-v2"
            or eligibility["status"] != "PASS-fixed-own64-bothseeds-eligible-no-formal-results"
            or eligibility["fixed_updates"] != 64
            or eligibility["seeds"] != list(SEEDS)
            or eligibility["formal_campaigns_used"] != 0
            or not 0 < spec["per_arm_seconds"] <= formal_duration(spec)
            or spec["reserved_output_bytes"] < 96 * 2**20):
        raise ValueError("eligible SINGLE fixed candidate before book selection")
    known = read(spec["known_protocol"])
    screen = read(spec["known160_audit"])
    if (screen["status"] != "PASS-fullhistory-integrity"
            or screen["screen_result"] != "PASS-development-only"
            or screen["games"] != 160
            or screen["protocol_sha256"] != spec["known_protocol"]["sha256"]
            or eligibility["models"] != known["models"]):
        raise ValueError("actual fixed candidate known160 and source eligibility")
    profile = read(spec["known48_profile"])
    if (profile["status"] != "PASS-ownNNUE64-parent-trained-profile-not-strength"
            or profile["protocol_sha256"] != spec["known_protocol"]["sha256"]
            or len(profile["packets"]) != 48
            or len(profile["parity"]) != 2
            or any(len(r["rows"]) != 24 for r in profile["parity"])
            or len(profile["latency_ratios"]) != 2
            or any(r["learned_vs_parent_median"] > 1.10
                   or r["learned_vs_E8_median"] > 1.10 for r in profile["latency_ratios"])):
        raise ValueError("actual compiled48/paired48 trained profile and unchanged latency")
    six = read(spec["known_six_replay"])
    if (six["status"] != "PASS-six-actual-fixed-chronological-search-packets"
            or len(six["packets"]) != 6 or six["finished"] > six["deadline"]
            or six["protocol_sha256"] != spec["known_protocol"]["sha256"]
            or six["profile_sha256"] != spec["known48_profile"]["sha256"]):
        raise ValueError("actual known160 six fixed chronological packet replay")
    exclusion = read(spec["exclusion_pass"])
    if (exclusion["schema"] != "ONE-ancestry-selected-DAG-book-exclusion-audit-v2"
            or exclusion["status"] != "PASS-zero-current-ancestry-and-known-book-overlap"
            or exclusion["candidate_roots"] != 96 or exclusion["matched_roots"] != 0
            or exclusion["books"] != spec["books"]
            or exclusion["eligibility_sha256"] != spec["eligibility"]["sha256"]
            or exclusion["current_DAG_coverage_complete"] is not True
            or exclusion["global_historical_coverage_claimed"] is not False):
        raise ValueError("complete current chosen ancestry exclusions; no global coverage fiction")
    read(exclusion["exposure_inventory"])
    verified_books = read(spec["book_verification"])
    if (verified_books["status"] !=
            "PASS96-fullhistory-draw-source-probability-and-current-DAG-exclusion"
            or verified_books["books"] != spec["books"]
            or verified_books["selection_sha256"] != spec["book_selection"]["sha256"]
            or verified_books["finished"] > verified_books["deadline"]):
        raise ValueError("actual independent bookselection/exclusion replay required")
    book_selection = read(spec["book_selection"])
    if (book_selection["eligibility_sha256"] != spec["eligibility"]["sha256"]
            or book_selection["selection_observed_epoch"] < eligibility["finished_epoch"]
            or book_selection["book_refs"] != spec["books"]
            or book_selection["outcomes_observed"] is not False
            or book_selection["independence_design"] != "independent-distinct-source-ECO-clusters"):
        raise ValueError("outcome-blind book selection after eligible candidate freeze")
    supplement = validate_supplement(
        spec["supplement"], spec["original_confirmation_DRAFT"], spec["user_v2_approval"],
        spec["current_experiment_preregistration"],
        sha(Path(__file__).with_name("bounded_betting.py")))
    first, deadline = spec["first"], spec["deadline"]
    if not (book_selection["selection_observed_epoch"] <= first
            <= time.time() < deadline <= spec["operator_end_epoch"]):
        raise ValueError("ROOT observed original formal clock; no future first/reset")
    if not deadline < spec["audit_deadline"] <= spec["operator_end_epoch"]:
        raise ValueError("bounded original audit within operator end")
    views = protocol_views(known, spec["books"])
    directory.mkdir(exist_ok=False)
    refs = {}
    for seed, q in views.items():
        q.update(original_first_epoch=first, original_deadline_epoch=deadline,
                 reserved_output_bytes=spec["reserved_output_bytes"],
                 ROOToperator_end_epoch=spec["operator_end_epoch"])
        for name, digest in spec["helper_sha256"].items():
            if sha(Path(__file__).with_name(name)) != digest:
                raise ValueError("entire formal helper closure pin")
        q["arena_helper_sha256"] = spec["helper_sha256"]
        path = directory / f"protocol-{seed}.json"
        publish(path, q)
        refs[seed] = dict(path=str(path), sha256=sha(path))
    registration = dict(spec, schema="ONE-NNUE-formal960-runtime-registration-v2",
                        status="registered-no-games-yet", views=refs,
                        supplemental_registration_binding=spec["supplement"],
                        supplement_original_registration_sha256=None,
                        prospective_supplement=supplement, total_games=960,
                        source_scope="ancestry conditional, not global virgin or generalElo")
    publish(directory / "registration.json", registration)
    binding = dict(schema="ONE-prospective-supplement-to-final-registration-binding-v1",
                   registration_sha256=sha(directory / "registration.json"),
                   supplement=spec["supplement"], parameter_changes=False,
                   binding_observed_epoch=time.time())
    publish(directory / "supplement-final-binding.json", binding)
    return registration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    seal(json.loads(a.spec.read_bytes()), a.output)


if __name__ == "__main__":
    main()
