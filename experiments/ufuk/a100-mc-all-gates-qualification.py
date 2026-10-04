"""Conjunctive method2 claim gate; no training, arenas or alternate selection."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def require_strength_gates(strength):
    assert [row["seed"] for row in strength["seed_results"]] == [20261205, 20261206]
    assert strength["replicated_strength_pass"] is True
    for row in strength["seed_results"]:
        assert row["strength_pass"] is True and all(row["gates"].values())
        assert row["direct"]["mean"] > 0.60 and row["direct"]["ci_adjusted_98_75"][0] > 0.50
        assert (
            row["sf_paired_delta"]["mean"] > 0.10
            and row["sf_paired_delta"]["ci_adjusted_98_75"][0] > 0
        )
        assert (
            row["final_sf_mean"] >= 0.25
            and set(row["caps"]) == {"direct", "final_sf", "initial_sf"}
            and all(value <= 0.05 for value in row["caps"].values())
        )
        assert (
            row["direct_adversarial_caps"]["ci_adjusted_98_75"][0] > 0.50
            and row["sf_delta_adversarial_caps"]["ci_adjusted_98_75"][0] > 0
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
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
    eligible = receipts["fixed_candidate_eligibility"]
    strength = receipts["strength_analysis"]
    sm = receipts["strength_manifest"]
    latency = receipts["latency"]
    assert eligible["status"] == "eligible-both-fixedCURRENT40-for-preregistered-strength-only"
    assert [row["seed"] for row in eligible["seeds"]] == [20261205, 20261206]
    assert [row["seed"] for row in sm["seeds"]] == [20261205, 20261206] and strength[
        "manifest_sha256"
    ] == m["strength_manifest"]["sha256"]
    require_strength_gates(strength)
    assert (
        strength["primary_comparisons"] == 4
        and strength["replicates"] == 50000
        and strength["two_sided_ci"] == 0.9875
    )
    assert strength["analysis_seed"] == 20261107 and strength["quantiles"] == "numpy linear"
    assert strength["analysis_sha256"] == sha(
        Path(__file__).with_name("a100-mc-strength-analysis.py")
    )
    model_shas = {str(row["seed"]): row["candidate_sha256"] for row in eligible["seeds"]}
    book_shas = {
        20261205: "b09f7dc1f4a8510c595c9732df195746f7a7455d1ccdc6ca8046301f70b3b0dc",
        20261206: "b790f63d616b5d54164e6d1358acffd3ae1dcf7f6c66f20865f9f09f0fadf70b",
    }
    for row in sm["seeds"]:
        assert any(
            candidate["seed"] == row["seed"] and candidate["source_commit"] == sm["source_commit"]
            for candidate in eligible["seeds"]
        )
        assert (
            row["final_sha256"] == model_shas[str(row["seed"])]
            and row["initial_sha256"]
            == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
        )
        assert sha(row["book"]) == book_shas[row["seed"]]
    assert (
        latency["speed_pass"] is True
        and latency["cpu_threads"] == 1
        and latency["warmup_rounds"] == 10
        and latency["timed_rounds"] == 200
        and latency["quiescent_declared"] is True
    )
    assert (
        latency["weights_sha256"]["initial"]
        == "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
    )
    for seed, digest in model_shas.items():
        assert (
            latency["weights_sha256"][seed] == digest and latency["ratios_vs_initial"][seed] <= 1.10
        )
    for name, backend in (("cuda_parity", "cuda"), ("mlx_cpu_parity", "mlx")):
        receipt = receipts[name]
        assert (
            receipt["status"] == "pass-sameweights-full-and-legalmasked18fullhistories"
            and receipt["backend"] == backend
            and receipt["atol"] == receipt["rtol"] == 2e-5
        )
        assert (
            receipt["probes_sha256"]
            == latency["probe_sha256"]
            == "1089fd0cca308c24bb040a840352d0ad18aa85a7bdcd17608133456bf08bed9f"
        )
        assert {str(row["seed"]): row["weights_sha256"] for row in receipt["models"]} == model_shas
    result = {
        "schema": "ufuk-method2-all-gates-conjunctive-qualification-v1",
        "status": "all-fixed-method2-gates-pass-on-two-frozen-source-root-suites",
        "manifest_sha256": sha(a.manifest),
        "script_sha256": sha(Path(__file__)),
        "receipt_sha256": {name: m[name]["sha256"] for name in receipts},
        "candidate_sha256": model_shas,
        "time_epoch": time.time(),
        "limits": (
            "Requires actual matching immutable receipts. Nominal bootstrap/IU"
            "T/MAX8-family interpretation; no general Elo or Stockfish-level c"
            "laim. Prospective ledger/protocol registration and external quies"
            "cence remain root-reviewed facts."
        ),
    }
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
