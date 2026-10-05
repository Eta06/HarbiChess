"""Read-only terminal scheduling proof; never turns incomplete family5 into qualification."""

import json
from pathlib import Path

from own6_failed4_dependency import identity_live, proc_table

SOURCE5 = "4515a7c0dda3b4f9615c2fc78a47c872ab14699d"
SEEDS5 = (20261525, 20261526)
END = 1791180000
ROLES = {
    "train-20261525",
    "train-20261526",
    "audit-20261525",
    "audit-20261526",
    "post-5",
    "baseline-20261525",
    "baseline-20261526",
}


def bound(path, digest, sha):
    assert sha(path) == digest
    return json.loads(Path(path).read_text())


def option(argv, name):
    assert argv.count(name) == 1
    return argv[argv.index(name) + 1]


def validate_terminal(terminal, sha):
    assert terminal["schema"] == "own5-incomplete-original-freshCLI600-terminal-v1"
    assert (
        terminal["status"] == "INCOMPLETE-original-freshCLI-replay-deadline-exhausted"
    )
    assert (
        terminal["qualification_ledger_slot"] == 5
        and terminal["source_commit"] == SOURCE5
    )
    config = bound(terminal["post_config"], terminal["post_config_sha256"], sha)
    assert (
        config["source_commit"] == SOURCE5 and config["root"] == terminal["post_root"]
    )
    assert [row["seed"] for row in config["seeds"]] == list(SEEDS5)
    assert all(
        row["original_training_started_epoch"]
        == terminal["original_training_started_epoch"]
        for row in config["seeds"]
    )
    q = bound(
        config["qualification_config"], config["qualification_config_sha256"], sha
    )
    assert q["source_commit"] == SOURCE5 and q["fixed_epochs"] == 8
    assert (
        terminal["original_qualification_config_sha256"]
        == config["qualification_config_sha256"]
    )
    supplement = bound(
        terminal["active_control_supplement"],
        terminal["active_control_supplement_sha256"],
        sha,
    )
    assert (
        supplement["schema"]
        == "own5-failed4-dependency-analysis-v3-control-supplement-v1"
    )
    assert supplement["new_config_sha256"] == terminal["post_config_sha256"]
    failure = bound(terminal["failure"], terminal["failure_sha256"], sha)
    assert Path(terminal["failure"]) == Path(config["root"]) / "failure.json"
    assert failure["status"] == "failed-or-incomplete-preserved-no-retry"
    assert (
        "deadline" in failure["error"].lower() and "exhaust" in failure["error"].lower()
    )
    command = bound(terminal["replay_command"], terminal["replay_command_sha256"], sha)
    seed = terminal["failed_replay_seed"]
    assert seed in SEEDS5
    assert (
        Path(terminal["replay_command"])
        == Path(config["root"]) / f"replay-{seed}-command.json"
    )
    assert command["deadline_epoch"] == command["started_epoch"] + 600
    assert command["deadline_epoch"] <= failure["finished_epoch"] < END
    assert terminal["expired_replay_deadline_epoch"] == command["deadline_epoch"]
    argv = command["argv"]
    assert Path(argv[1]).name == "own5_fresh_cli_replay.py"
    row = next(row for row in config["seeds"] if row["seed"] == seed)
    assert option(argv, "--run") == row["run"]
    assert option(argv, "--manifest") == row["audit_manifest"]
    assert sha(row["audit_manifest"]) == option(argv, "--manifest-sha256")
    inner = float(option(argv, "--deadline-epoch"))
    assert 0 <= command["deadline_epoch"] - inner <= 1
    assert terminal["method5_replay_retry_or_budget_reset"] is False
    identities = terminal["owner_identities"]
    assert len({row["pid"] for row in identities if "pid" in row}) == sum(
        "pid" in row for row in identities
    )
    assert {row["role"] for row in identities} >= ROLES
    for row in identities:
        if "pid" in row:
            assert type(row["pid"]) is int and row["pid"] > 0
            assert type(row["startticks"]) is int and row["startticks"] > 0
        else:
            ended = bound(
                row["terminated_owner_receipt"],
                row["terminated_owner_receipt_sha256"],
                sha,
            )
            assert type(ended["returncode"]) is int
            assert ended["finished_epoch"] <= terminal["observed_epoch"] < END
    return config


def validate_release(release, terminal, terminal_sha256, sha, table=None):
    validate_terminal(terminal, sha)
    assert release["schema"] == "own5-incomplete-owned-compute-terminated-witness-v1"
    assert release["terminal_receipt_sha256"] == terminal_sha256
    assert release["all_registered_owned_groups_and_identities_terminated"] is True
    assert release["method5_ready_latency_final_receipts_fabricated"] is False
    assert release["observed_epoch"] >= terminal["expired_replay_deadline_epoch"]
    assert release["observed_epoch"] < END
    # Parent-owned before/after inventories are immutable evidence, not a stale PID list.
    before = bound(release["before_inventory"], release["before_inventory_sha256"], sha)
    after = bound(release["after_inventory"], release["after_inventory_sha256"], sha)
    assert before["source_commit"] == after["source_commit"] == SOURCE5
    assert (
        before["qualification_ledger_slot"] == after["qualification_ledger_slot"] == 5
    )
    assert (
        before["terminal_receipt_sha256"]
        == after["terminal_receipt_sha256"]
        == terminal_sha256
    )
    assert after["remaining_owned_pid_startticks"] == []
    registered = {
        (row["pid"], row["startticks"])
        for row in terminal["owner_identities"]
        if "pid" in row
    }
    tracked = release["terminated_tracked_pid_startticks"]
    tracked_pairs = {(row["pid"], row["startticks"]) for row in tracked}
    assert registered <= tracked_pairs
    assert tracked_pairs == {
        (row["pid"], row["startticks"])
        for row in before["tracked_owned_pid_startticks"]
    }
    actual = proc_table() if table is None else table
    assert not any(identity_live(row, actual) for row in tracked)
    groups = set(before["owned_process_group_ids"])
    assert not any(
        row["state"] not in ("Z", "X") and row["pgid"] in groups
        for row in actual.values()
    )
    return {
        "status": "prior-family5-incomplete-and-owned-compute-ended-scheduling-only",
        "terminal_receipt_sha256": terminal_sha256,
        "release_receipt_sha256": release.get("receipt_sha256"),
        "qualification_or_latency_or_final_receipt_substituted": False,
    }
