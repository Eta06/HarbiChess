import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import failed5_terminal_barrier as b


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def witness(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return str(path), sha(path)

    root = tmp_path / "post5"
    root.mkdir()
    qpath, qsha = write("Q.json", {"source_commit": b.SOURCE5, "fixed_epochs": 8})
    mpath, msha = write("audit25.json", {"source_commit": b.SOURCE5})
    rows = [
        {
            "seed": seed,
            "original_training_started_epoch": 1791156456.0081534,
            "run": f"/content/runs/seed-{seed}/run",
            "audit_manifest": mpath,
        }
        for seed in b.SEEDS5
    ]
    cpath, csha = write(
        "post.json",
        {
            "source_commit": b.SOURCE5,
            "root": str(root),
            "seeds": rows,
            "qualification_config": qpath,
            "qualification_config_sha256": qsha,
        },
    )
    spath, ssha = write(
        "supplement.json",
        {
            "schema": "own5-failed4-dependency-analysis-v3-control-supplement-v1",
            "new_config_sha256": csha,
        },
    )
    first = 1791162251.89774
    command = {
        "started_epoch": first,
        "deadline_epoch": first + 600,
        "argv": [
            "/usr/bin/python3",
            "/helpers/own5_fresh_cli_replay.py",
            "--run",
            rows[0]["run"],
            "--manifest",
            mpath,
            "--manifest-sha256",
            msha,
            "--deadline-epoch",
            str(first + 599.999),
        ],
    }
    cp = root / "replay-20261525-command.json"
    cp.write_text(json.dumps(command))
    fp = root / "failure.json"
    fp.write_text(
        json.dumps(
            {
                "status": "failed-or-incomplete-preserved-no-retry",
                "error": "TimeoutError('original wholedeadline exhausted')",
                "finished_epoch": first + 600.002,
            }
        )
    )
    identities = [
        {"role": role, "pid": 100 + i, "startticks": 1000 + i}
        for i, role in enumerate(sorted(b.ROLES))
    ]
    terminal = {
        "schema": "own5-incomplete-original-freshCLI600-terminal-v1",
        "status": "INCOMPLETE-original-freshCLI-replay-deadline-exhausted",
        "qualification_ledger_slot": 5,
        "source_commit": b.SOURCE5,
        "post_config": cpath,
        "post_config_sha256": csha,
        "post_root": str(root),
        "original_training_started_epoch": rows[0]["original_training_started_epoch"],
        "original_qualification_config_sha256": qsha,
        "active_control_supplement": spath,
        "active_control_supplement_sha256": ssha,
        "failure": str(fp),
        "failure_sha256": sha(fp),
        "replay_command": str(cp),
        "replay_command_sha256": sha(cp),
        "failed_replay_seed": b.SEEDS5[0],
        "expired_replay_deadline_epoch": first + 600,
        "method5_replay_retry_or_budget_reset": False,
        "owner_identities": identities,
    }
    tpath, tsha = write("terminal.json", terminal)
    before = {
        "source_commit": b.SOURCE5,
        "qualification_ledger_slot": 5,
        "terminal_receipt_sha256": tsha,
        "tracked_owned_pid_startticks": identities,
        "owned_process_group_ids": [100],
    }
    after = {
        "source_commit": b.SOURCE5,
        "qualification_ledger_slot": 5,
        "terminal_receipt_sha256": tsha,
        "remaining_owned_pid_startticks": [],
    }
    pre, presha = write("before.json", before)
    post, postsha = write("after.json", after)
    release = {
        "schema": "own5-incomplete-owned-compute-terminated-witness-v1",
        "terminal_receipt_sha256": tsha,
        "all_registered_owned_groups_and_identities_terminated": True,
        "method5_ready_latency_final_receipts_fabricated": False,
        "observed_epoch": first + 610,
        "before_inventory": pre,
        "before_inventory_sha256": presha,
        "after_inventory": post,
        "after_inventory_sha256": postsha,
        "terminated_tracked_pid_startticks": identities,
    }
    return terminal, release, tsha


def test_original600_failure_then_owned_release_is_schedule_only(witness):
    terminal, release, tsha = witness
    result = b.validate_release(release, terminal, tsha, sha, {})
    assert result["qualification_or_latency_or_final_receipt_substituted"] is False


@pytest.mark.parametrize(
    "mutation", ["budget-reset", "failure-clock", "foreign-source", "missing-owner"]
)
def test_terminal_mutations_rejected(witness, mutation):
    terminal, _, _ = witness
    terminal = copy.deepcopy(terminal)
    if mutation == "budget-reset":
        terminal["method5_replay_retry_or_budget_reset"] = True
    if mutation == "failure-clock":
        terminal["expired_replay_deadline_epoch"] += 600
    if mutation == "foreign-source":
        terminal["source_commit"] = "428a30f5e1658f3cf159844db547ff0147ade5a9"
    if mutation == "missing-owner":
        terminal["owner_identities"].pop()
    with pytest.raises(AssertionError):
        b.validate_terminal(terminal, sha)


def test_live_same_owner_and_new_session_child_block_release(witness):
    terminal, release, tsha = witness
    owner = terminal["owner_identities"][0]
    table = {
        owner["pid"]: {"startticks": owner["startticks"], "state": "S", "pgid": 999}
    }
    with pytest.raises(AssertionError):
        b.validate_release(release, terminal, tsha, sha, table)
    # Unregistered old group member blocks even if owner has exited.
    with pytest.raises(AssertionError):
        b.validate_release(
            release,
            terminal,
            tsha,
            sha,
            {999: {"startticks": 9, "state": "S", "pgid": 100}},
        )


def test_reused_pid_not_old_owner_and_zombie_are_noncompute(witness):
    terminal, release, tsha = witness
    owner = terminal["owner_identities"][0]
    table = {
        owner["pid"]: {"startticks": owner["startticks"] + 1, "state": "S", "pgid": 999}
    }
    b.validate_release(release, terminal, tsha, sha, table)
    table[owner["pid"]] = {"startticks": owner["startticks"], "state": "Z", "pgid": 100}
    b.validate_release(release, terminal, tsha, sha, table)


def test_changed_inventory_sha_and_fake_latency_rejected(witness):
    terminal, release, tsha = witness
    corrupt = copy.deepcopy(release)
    corrupt["method5_ready_latency_final_receipts_fabricated"] = True
    with pytest.raises(AssertionError):
        b.validate_release(corrupt, terminal, tsha, sha, {})
    Path(release["after_inventory"]).write_text("{}")
    with pytest.raises(AssertionError):
        b.validate_release(release, terminal, tsha, sha, {})


def test_launcher_checks_failed5_and_failed4_release_before_new900_clock():
    import ast

    tree = ast.parse((Path(__file__).parent / "launch_after45_terminal.py").read_text())
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    assert any(
        isinstance(n.func, ast.Name) and n.func.id == "validate_barrier" for n in calls
    )
    text = ast.unparse(main)
    assert text.index("proof = validate_barrier(config)") < text.index(
        "first = time.time()"
    )
    assert "deadline = first + 900" in text
    assert "cohort-latency-complete.json" not in text
    assert "latest_profile_start_epoch" in text


def test_actual_proc_snapshot_supports_old_owned_group_validator():
    import importlib.util

    path = Path(__file__).parent / "launch_after45_terminal.py"
    spec = importlib.util.spec_from_file_location("terminal_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    from failed4_dependency import remaining_owned

    table = launcher.process_table()
    assert table and all("pgid" in row for row in table.values())
    assert (
        remaining_owned({"owned_process_group_ids": [], "owner_identities": []}, table)
        == []
    )
