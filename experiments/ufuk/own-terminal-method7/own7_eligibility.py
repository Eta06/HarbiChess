"""Fail-closed prospective OWN-v1 eligibility; requires real new-schema independent audit."""

import argparse
import hashlib
import json
from pathlib import Path

from own7_adapter_controls import SOURCE

SEEDS = (20261725, 20261726)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.manifest)
    assert manifest["qualification_ledger_slot"] == 7
    assert manifest["source_commit"] == SOURCE
    assert [r["seed"] for r in manifest["seeds"]] == list(SEEDS)
    epochs = manifest["fixed_epochs"]
    assert type(epochs) is int and epochs >= 8
    results = []
    for row in manifest["seeds"]:
        run = Path(row["run"])
        native = run / "checkpoints" / f"epoch-{epochs:08d}"
        checkpoint = read(native / "checkpoint.json")
        state = read(native / "actor.json")
        assert checkpoint["schema"] == "torch-search-acting-native-cuda-v2"
        assert checkpoint["run_config"]["schema"] == "torch-fresh-sparse-search-acting-v2"
        assert (
            checkpoint["source_commit"] == manifest["source_commit"]
            and checkpoint["state"] == state
        )
        assert state["epoch"] == epochs and state["fresh_transitions"] == 32768 * epochs
        assert state["actor_steps"] == 256 * epochs and len(state["search_rngs"]) == 128
        assert state["pending_search_schedule"] == state["replay_buffer"] == "closed-empty"
        assert set(checkpoint["artifacts"]) == {
            "model.safetensors",
            "base.safetensors",
            "behavior.safetensors",
            "training.pt",
            "actor.json",
            "last-frozen-epoch.json.gz",
        }
        assert state["optimizer_accepted_updates"] > 0
        for filename, digest in checkpoint["artifacts"].items():
            assert sha(native / filename) == digest
        full, replay = (
            read(row[name]["path"])
            for name in ("full_search_acting_audit", "fresh_search_acting_replay")
        )
        for name in ("full_search_acting_audit", "fresh_search_acting_replay"):
            assert sha(row[name]["path"]) == row[name]["sha256"]
        assert full["schema"] == "ufuk-search-acting-allnative-data-audit-v3"
        assert full["status"] == "pass-all-fixed-method7-native-own-data-certificate-search-ledgers"
        assert full["source_commit"] == replay["source_commit"] == manifest["source_commit"]
        assert full["seed"] == replay["seed"] == row["seed"]
        assert full["fixed_epochs"] == epochs
        assert full["finished_epoch"] <= full["deadline_epoch"]
        assert full["audited_native_checkpoints"] == epochs + 1
        assert full["independently_replayed_fresh_transitions"] == epochs * 32768
        assert full["prescribed_neural_roots_verified"] == 108
        assert full["raw_actor_packet_roots_verified"] == 108
        assert full["neural_search_roots_recomputed"] >= 108
        assert full["schedule_and_all128search_rngs_verified"] is True
        assert full["actual_mixed_mu_replayed"] is True
        assert full["raw_policy_KL_reference"] == "KL(frozen-raw-pi||current-raw-pi)"
        assert full["neural_witness_K"] == 8
        assert full["sample_chain_sha256"] == state["sample_chain_sha256"]
        assert full["optimizer_updates_performed_by_this_audit"] == 0
        assert replay["schema"] == "ufuk-search-acting-independent-freshCLI-replay-v3"
        assert (
            replay["status"] == "pass-exact-search-acting-v3-native2-all-six-payloads-and-journal"
        )
        assert replay["duplicate_fresh_presentations"] == 32768
        metadata = read(run / "metadata.json")
        assert metadata["schema"] == "search-acting-supervised-run-v2"
        assert replay["original_deadline_epoch"] == metadata["absolute_deadline_epoch"]
        assert (
            replay["finished_epoch"]
            <= replay["audit_deadline_epoch"]
            <= replay["started_epoch"] + 600
        )
        expected_compared = {"journal/epoch-00000002.json.gz"} | {
            "checkpoints/epoch-00000002/" + filename for filename in checkpoint["artifacts"]
        }
        assert set(replay["compared_sha256"]) == expected_compared
        for relative, digest in replay["compared_sha256"].items():
            assert sha(run / relative) == digest
        assert full["manifest_sha256"] == replay["manifest_sha256"]
        assert full["audit_sha256"] == manifest["audit_helper_sha256"]
        assert replay["script_sha256"] == manifest["replay_helper_sha256"]
        assert Path(replay["original_run"]).resolve() == run.resolve()
        assert len(full["immutable_percheckpoint_audit_receipt_sha256"]) == epochs + 1
        for path, digest in full["immutable_percheckpoint_audit_receipt_sha256"].items():
            assert sha(path) == digest
        results.append(
            {
                "seed": row["seed"],
                "epoch": epochs,
                "source_commit": manifest["source_commit"],
                "candidate_sha256": sha(native / "model.safetensors"),
                "native_manifest_sha256": sha(native / "checkpoint.json"),
            }
        )
    result = {
        "schema": "ufuk-certificate-search-fixed-candidate-eligibility-v3",
        "qualification_ledger_slot": 7,
        "status": "eligible-both-fixedMETHOD7-native-v2-ledger-v3-for-preregistered-strength-only",
        "fixed_epochs": epochs,
        "seeds": results,
        "manifest_sha256": sha(args.manifest),
        "script_sha256": sha(Path(__file__)),
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
