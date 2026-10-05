"""Prospective standalone8 barrier; preserved actual6/7 failures and owned release."""

import json
import time
from pathlib import Path

ADMISSION = 1791169800
PRELATENCY = 1791174900
LATEST_LATENCY_START = 1791175200
END = 1791180000


def validate_descriptor(config, sha):
    path = Path(config["manifest"])
    assert sha(path) == config["manifest_sha256"]
    d = json.loads(path.read_text())
    assert d["schema"] == "prospective-own8-standalone-prior-failures-v1"
    assert d["status"] == "frozen-before-any-method8-compute"
    assert d["qualification_ledger_slot"] == 8 and d["seeds"] == [20261825, 20261826]
    assert d["admission_deadline_epoch"] == ADMISSION
    assert d["prelatency_deadline_epoch"] == PRELATENCY
    assert d["latest_latency_start_epoch"] == LATEST_LATENCY_START
    assert d["hard_deadline_epoch"] == END
    assert set(d["prior_unadmitted_failure"]) == {"6", "7"}
    from own8_terminal45_release import process_table

    table = process_table()
    six = d["prior_unadmitted_failure"]["6"]
    assert (
        sha(six["terminal"])
        == six["terminal_sha256"]
        == "5290964aea5366cf233f60ada032919c1c37e07f377fdbe704f959844298205b"
    )
    terminal = json.loads(Path(six["terminal"]).read_text())
    assert (
        terminal["slot"] == 6
        and terminal["status"]
        == "INCOMPLETE-no-qualification-or-early-candidate-selection"
    )
    assert terminal["original_clocks_or_budgets_reset"] is False
    assert terminal["ready_latency_final_receipts_fabricated"] is False
    assert (
        terminal["actual_failure_sha256"]
        == "508c8b7a050b91878eb03d75023f638dd59b37868a5a3055450430d42d4ca5f5"
    )
    for name in ("actual_failure", "before_inventory", "after_inventory"):
        assert sha(terminal[name]) == terminal[name + "_sha256"]
    before = json.loads(Path(terminal["before_inventory"]).read_text())
    after = json.loads(Path(terminal["after_inventory"]).read_text())
    assert after["remaining_owned_pid_startticks"] == []
    assert (
        terminal["terminated_tracked_pid_startticks"]
        == before["tracked_owned_pid_startticks"]
    )
    for row in before["tracked_owned_pid_startticks"]:
        assert not (
            table.get(row["pid"], {}).get("startticks") == row["startticks"]
            and table[row["pid"]]["state"] not in ("Z", "X")
        )
    assert not any(
        row["state"] not in ("Z", "X")
        and row["pgid"] in before["owned_process_group_ids"]
        for row in table.values()
    )
    seven = d["prior_unadmitted_failure"]["7"]
    assert (
        sha(seven["original_failure"])
        == seven["original_failure_sha256"]
        == "7282af0b37f2da04d2686e52ef4aef10287ec1562a37db5bc5252ba853a196a2"
    )
    failed = json.loads(Path(seven["original_failure"]).read_text())
    assert (
        failed["status"] == "failed-preserved"
        and failed["source_commit"] == "c022bc1605b44c3089439da5c6efb7bd4db4ff81"
    )
    assert (
        failed["absolute_deadline_epoch"] == 1791166813.4247744
        and failed["finished_epoch"] <= failed["absolute_deadline_epoch"]
    )
    assert (
        sha(seven["actual_owned_inventory"]) == seven["actual_owned_inventory_sha256"]
    )
    inventory = json.loads(Path(seven["actual_owned_inventory"]).read_text())
    assert (
        inventory["files"]["result.json"]["sha256"] == seven["original_failure_sha256"]
    )
    assert json.loads(inventory["files"]["result.json"]["text"]) == failed
    assert inventory["observed_epoch"] >= failed["finished_epoch"]
    assert (
        inventory["processes"] == {"125937": None, "128081": None}
        and inventory["groups"] == {}
    )
    assert (
        seven["formal_training_started"] is False
        and seven["formal_admission_published"] is False
    )
    # New descriptor attests only never-admission, not invented readiness/latency.
    return d


def wait_previous(config, wait_json, sha):
    del wait_json
    assert time.time() < PRELATENCY
    from own8_schedule_v3 import verify_previous

    previous45 = verify_previous(config["previous45_completion_barrier"], sha)
    descriptor = validate_descriptor(config["standalone8_prior_failures"], sha)
    return dict(previous45=previous45, prior67=descriptor)
