"""Conjunctive SEARCH-ACTING-v2 method5 claim gate; no training, arenas or alternate selection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from analysis_repair_binding import bind_cli
from own5_strength_config import BOOKS, SEEDS, validate_binding

Q = {}
REPAIR_SHA = None
ORIGINAL_Q_SHA = None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def require_strength_gates(strength):
    assert [row["seed"] for row in strength["seed_results"]] == list(SEEDS)
    assert strength["replicated_strength_pass"] is True
    for row in strength["seed_results"]:
        assert row["strength_pass"] is True and all(row["gates"].values())
        assert (
            row["direct"]["mean"] > 0.6 and row["direct"]["ci_adjusted_98_75"][0] > 0.5
        )
        assert (
            row["sf_paired_delta"]["mean"] > 0.1
            and row["sf_paired_delta"]["ci_adjusted_98_75"][0] > 0
        )
        assert (
            row["final_sf_mean"] >= 0.25
            and set(row["caps"]) == {"direct", "final_sf", "initial_sf"}
            and all(value <= 0.05 for value in row["caps"].values())
        )
        assert (
            row["direct_adversarial_caps"]["ci_adjusted_98_75"][0] > 0.5
            and row["sf_delta_adversarial_caps"]["ci_adjusted_98_75"][0] > 0
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    validate_binding(a, Q)
    m = read(a.manifest)
    receipts = {}
    for name in (
        "fixed_candidate_eligibility",
        "strength_analysis",
        "strength_manifest",
        "latency",
        "cuda_parity",
        "mlx_cpu_parity",
    ):
        info = m[name]
        assert sha(info["path"]) == info["sha256"]
        receipts[name] = read(info["path"])
    for name in ("strength_analysis", "latency", "cuda_parity", "mlx_cpu_parity"):
        assert receipts[name]["source_commit"] == Q["source_commit"]
        assert receipts[name]["fixed_epochs"] == Q["fixed_epochs"]
        assert receipts[name]["qualification_ledger_slot"] == 5
    eligible = receipts["fixed_candidate_eligibility"]
    assert eligible["schema"] == "ufuk-search-acting-fixed-candidate-eligibility-v2"
    assert eligible["fixed_epochs"] == Q["fixed_epochs"]
    assert all(
        row["epoch"] == Q["fixed_epochs"] and row["source_commit"] == Q["source_commit"]
        for row in eligible["seeds"]
    )
    strength = receipts["strength_analysis"]
    assert strength["analysis_repair_supplement_sha256"] == REPAIR_SHA
    assert strength["original_qualification_config_sha256"] == ORIGINAL_Q_SHA
    sm = receipts["strength_manifest"]
    assert (
        eligible["qualification_ledger_slot"]
        == strength["qualification_ledger_slot"]
        == sm["qualification_ledger_slot"]
        == 5
    )
    latency = receipts["latency"]
    assert (
        eligible["status"]
        == "eligible-both-fixedSEARCH_ACTINGv2-for-preregistered-strength-only"
    )
    assert [row["seed"] for row in eligible["seeds"]] == list(SEEDS)
    assert [row["seed"] for row in sm["seeds"]] == list(SEEDS) and strength[
        "manifest_sha256"
    ] == m["strength_manifest"]["sha256"]
    require_strength_gates(strength)
    assert (
        strength["primary_comparisons"] == 5
        and strength["replicates"] == 50000
        and (strength["two_sided_ci"] == 0.9875)
    )
    assert (
        strength["analysis_seed"] == 20261107
        and strength["quantiles"] == "numpy linear"
    )
    assert strength["analysis_sha256"] == sha(
        Path(__file__).with_name("own5_strength_analysis.py")
    )
    model_shas = {
        str(row["seed"]): row["candidate_sha256"] for row in eligible["seeds"]
    }
    for row in sm["seeds"]:
        assert any(
            candidate["seed"] == row["seed"]
            and candidate["source_commit"] == sm["source_commit"]
            for candidate in eligible["seeds"]
        )
        assert (
            row["final_sha256"] == model_shas[str(row["seed"])]
            and row["initial_sha256"]
            == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
        )
        assert sha(row["book"]) == BOOKS[row["seed"]]
    assert (
        latency["speed_pass"] is True
        and latency["cpu_threads"] == 1
        and (latency["warmup_rounds"] == 10)
        and (latency["timed_rounds"] == 200)
        and (latency["quiescent_declared"] is True)
    )
    assert (
        latency["weights_sha256"]["initial"]
        == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    )
    for seed, digest in model_shas.items():
        assert (
            latency["weights_sha256"][seed] == digest
            and latency["ratios_vs_initial"][seed] <= 1.1
        )
    for name, backend in (("cuda_parity", "cuda"), ("mlx_cpu_parity", "mlx")):
        receipt = receipts[name]
        assert (
            receipt["status"] == "pass-sameweights-full-and-legalmasked18fullhistories"
            and receipt["backend"] == backend
            and (receipt["atol"] == receipt["rtol"] == 2e-05)
        )
        assert (
            receipt["probes_sha256"]
            == latency["probe_sha256"]
            == "1089fd0cca308c24bb040a840352d0ad18aa85a7bdcd17608133456bf08bed9f"
        )
        assert {
            str(row["seed"]): row["weights_sha256"] for row in receipt["models"]
        } == model_shas
    result = {
        "schema": "ufuk-own5-all-gates-conjunctive-qualification-v1",
        "qualification_ledger_slot": 5,
        "status": "all-fixed-SEARCH_ACTINGv2-gates-pass-on-two-frozen-source-root-suites",
        "manifest_sha256": sha(a.manifest),
        "script_sha256": sha(Path(__file__)),
        "receipt_sha256": {name: m[name]["sha256"] for name in receipts},
        "candidate_sha256": model_shas,
        "time_epoch": time.time(),
        "limits": (
            "Requires actual matching immutable receipts. Nominal bootstrap/I"
            "UT/MAX8-family interpretation; no general Elo or Stockfish-level"
            " claim. Prospective ledger/protocol registration and external qu"
            "iescence remain root-reviewed facts."
        ),
    }
    result.update(
        source_commit=Q["source_commit"],
        fixed_epochs=Q["fixed_epochs"],
        qualification_ledger_slot=5,
    )
    result.update(
        analysis_repair_supplement_sha256=REPAIR_SHA,
        original_qualification_config_sha256=ORIGINAL_Q_SHA,
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    bind_cli(globals())
    main()
