"""Read completion metadata only; never change the original method4/5 cohort."""

import json
from pathlib import Path

from own6_failed4_dependency import (
    identity_live,
    proc_table,
    remaining_owned,
    validate_terminal,
)

LATEST_LATENCY_START = 1791171600  # 2026-10-05 03:40 UTC
SOURCES = {
    4: "a278bba67bce962cb9294f0d24040e02e9acf1f4",
    5: "4515a7c0dda3b4f9615c2fc78a47c872ab14699d",
}
SEEDS = {4: (20261425, 20261426), 5: (20261525, 20261526)}


def repaired5_member(info, cohort, sha):
    supplement_path = Path(info["failed4_dependency_supplement"])
    assert sha(supplement_path) == info["failed4_dependency_supplement_sha256"]
    supplement = json.loads(supplement_path.read_text())
    assert (
        supplement["schema"]
        == "own5-failed4-dependency-analysis-v3-control-supplement-v1"
    )
    assert supplement["qualification_ledger_slot"] == 5
    assert sha(supplement["new_config"]) == supplement["new_config_sha256"]
    config = json.loads(Path(supplement["new_config"]).read_text())
    assert config["cohort"]["manifest_sha256"] == info["original_cohort_sha256"]
    assert (
        sha(supplement["failed4_terminal_receipt"])
        == supplement["failed4_terminal_receipt_sha256"]
    )
    terminal = json.loads(Path(supplement["failed4_terminal_receipt"]).read_text())
    validate_terminal(terminal, cohort, sha)
    member = dict(next(m for m in cohort["members"] if m["slot"] == 5))
    member["root"] = config["root"]
    return supplement, terminal, member


def verify_failed4_release(path, release, supplement, terminal, sha):
    assert release["schema"] == "own4-incomplete-owned-compute-terminated-witness-v1"
    assert (
        release["terminal_receipt_sha256"]
        == supplement["failed4_terminal_receipt_sha256"]
    )
    assert release["all_registered_owned_groups_and_identities_terminated"] is True
    assert release["method4_ready_latency_final_receipts_fabricated"] is False
    assert release["control_supplement_sha256"] == sha(path)
    table = proc_table()
    assert not remaining_owned(terminal, table)
    assert not any(
        identity_live(item, table)
        for item in release["terminated_tracked_pid_startticks"]
    )


def wait_previous(config, wait_json, sha):
    info = config["previous45_completion_barrier"]
    cohort_path = Path(info["original_cohort"])
    assert sha(cohort_path) == info["original_cohort_sha256"]
    cohort = wait_json(cohort_path, LATEST_LATENCY_START)
    assert cohort["slots"] == [4, 5] and cohort["latency_order"] == [4, 5]
    proof = {}
    members = cohort["members"]
    if "failed4_dependency_supplement" in info:
        supplement, terminal, only5 = repaired5_member(info, cohort, sha)
        release_path = Path(only5["root"]) / "failed4-dependency-release.json"
        release = wait_json(release_path, LATEST_LATENCY_START)
        verify_failed4_release(
            info["failed4_dependency_supplement"], release, supplement, terminal, sha
        )
        assert release["observed_epoch"] <= cohort["completion_deadline_epoch"]
        proof[str(release_path)] = sha(release_path)
        proof[supplement["failed4_terminal_receipt"]] = supplement[
            "failed4_terminal_receipt_sha256"
        ]
        members = [only5]
    for member in members:
        slot = member["slot"]
        assert member["source_commit"] == SOURCES[slot]
        root = Path(member["root"])
        receipt_path = root / "cohort-latency-complete.json"
        receipt = wait_json(receipt_path, LATEST_LATENCY_START)
        assert receipt["schema"] == "own45-latency-owner-completion-receipt-v1"
        assert receipt["slot"] == slot and receipt["source_commit"] == SOURCES[slot]
        assert receipt["coordinator_sha256"] == member["coordinator_sha256"]
        assert receipt["finished_epoch"] <= cohort["completion_deadline_epoch"]
        assert set(receipt["process_receipt_sha256"]) == {"latency-process-result.json"}
        for filename, digest in receipt["process_receipt_sha256"].items():
            path = root / filename
            assert sha(path) == digest
            result = wait_json(path, LATEST_LATENCY_START)
            assert result["returncode"] == 0
            assert result["finished_epoch"] <= result["deadline_epoch"]
            proof[str(path)] = digest
        proof[str(receipt_path)] = sha(receipt_path)
        for seed in SEEDS[slot]:
            path = root / f"final-{seed}-process-result.json"
            result = wait_json(path, LATEST_LATENCY_START)
            assert result["returncode"] == 0
            assert result["finished_epoch"] <= result["deadline_epoch"]
            proof[str(path)] = sha(path)
    return proof
