"""Bind BOTH immutable CURRENT40 candidates to completed production/audits.

Passing this check authorizes no training or promotion; receipt only establishes
fixed candidate eligibility for already preregistered heldout strength games.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    m = read(a.manifest)
    assert m["qualification_ledger_slot"] == 3
    assert [row["seed"] for row in m["seeds"]] == [20261205, 20261206]
    out = []
    for row in m["seeds"]:
        run = Path(row["run"])
        meta = read(run / "metadata.json")
        source = m["source_commit"]
        seed = row["seed"]
        assert (
            meta["source_commit"] == source
            and meta["config"]["seed"] == seed
            and meta["max_epochs"] == 40
            and meta["checkpoint_interval"] == 1
        )
        native = run / "checkpoints/epoch-00000040"
        checkpoint = read(native / "checkpoint.json")
        state = read(native / "actor.json")
        assert (
            checkpoint["source_commit"] == source
            and checkpoint["state"] == state
            and state["epoch"] == 40
            and state["actor_steps"] == 10240
            and state["fresh_transitions"] == 1310720
            and state["replay_buffer"] == "closed-empty"
            and state["optimizer_accepted_updates"] > 0
        )
        current = Path(row["candidate"])
        assert (
            sha(current)
            == sha(native / "model.safetensors")
            == checkpoint["artifacts"]["model.safetensors"]
        )
        receipt_rows = []
        for name in ("full_native_audit", "fresh_replay_audit"):
            info = row[name]
            path = Path(info["path"])
            assert sha(path) == info["sha256"]
            receipt_rows.append(read(path))
        full, replay = receipt_rows
        assert full["audit_sha256"] == sha(
            Path(__file__).with_name("a100-mc-native-terminal-audit-v3-incremental.py")
        )
        assert replay["source_commit"] == source
        assert replay["script_sha256"] == sha(
            Path(__file__).with_name("a100-mc-fresh-cli-replay.py")
        )
        assert replay["manifest_sha256"] == full["manifest_sha256"]
        assert (
            full["status"] == "pass-all41native-all40epochs-independent-own-data"
            and full["source_commit"] == source
            and full["seed"] == seed
            and full["sample_chain_sha256"] == state["sample_chain_sha256"]
            and full["audited_native_checkpoints"] == 41
            and full["independently_replayed_fresh_transitions"] == 1310720
        )
        assert full["checkpoints"][-1]["checkpoint_manifest_sha256"] == sha(
            native / "checkpoint.json"
        )
        assert full["neural_audit_epochs"] == [1, 2, 10, 20, 30, 40]
        assert [receipt["epoch"] for receipt in full["neural_receipt_validation"]] == [
            1,
            2,
            10,
            20,
            30,
            40,
        ]
        assert all(
            receipt["samples"] == 18 and receipt["atol"] == receipt["rtol"] == 2e-5
            for receipt in full["neural_receipt_validation"]
        )
        assert (
            replay["status"] == "pass-exact-freshCLI-epoch2-allnative-payloadbits"
            and Path(replay["original_run"]).resolve() == run.resolve()
            and replay["original_deadline_epoch"] == meta["absolute_deadline_epoch"]
            and replay["duplicate_fresh_presentations"] == 32768
        )
        for relative, digest in replay["compared_sha256"].items():
            assert sha(run / relative) == digest
        results = [read(path) for path in sorted(run.glob("invocation-*-result.json"))]
        completed = [r for r in results if r["status"] == "completed"]
        assert any(
            r["epoch"] == 1
            and r["closed_boundary"] is True
            and r["finished_epoch"] <= meta["absolute_deadline_epoch"]
            for r in completed
        )
        assert any(
            r["epoch"] == 40
            and r["closed_boundary"] is True
            and r["finished_epoch"] <= meta["absolute_deadline_epoch"]
            and r["sample_chain_sha256"] == state["sample_chain_sha256"]
            for r in completed
        )
        commands = [read(path) for path in sorted(run.glob("invocation-*-command.json"))]
        assert any(
            c["resume"]
            and Path(c["resume"]).resolve() == (run / "checkpoints/epoch-00000001").resolve()
            for c in commands
        )
        out.append(
            {
                "seed": seed,
                "candidate_path": str(current.resolve()),
                "candidate_sha256": sha(current),
                "epoch": 40,
                "source_commit": source,
                "native_manifest_sha256": sha(native / "checkpoint.json"),
                "audit_receipt_sha256": {
                    name: row[name]["sha256"]
                    for name in ("full_native_audit", "fresh_replay_audit")
                },
            }
        )
    result = {
        "schema": "ufuk-method2-fixed-candidate-eligibility-v1",
        "qualification_ledger_slot": 3,
        "training_lineage": (
            "Original formal2 fixedCURRENT40 bothseeds; no training restart or native input rewrite"
        ),
        "status": "eligible-both-fixedCURRENT40-for-preregistered-strength-only",
        "seeds": out,
        "manifest_sha256": sha(a.manifest),
        "script_sha256": sha(Path(__file__)),
        "time_epoch": time.time(),
        "scope": (
            "No strength gain or model promotion. Quiescent latency and all fo"
            "ur inferential/both-seed strength gates remain required."
        ),
    }
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
