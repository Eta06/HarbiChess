import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cohort67", HERE / "own67_cohort.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path), c.sha(path)


@pytest.fixture
def cohort(tmp_path):
    members = []
    for slot in (6, 7):
        root = tmp_path / f"root{slot}"
        root.mkdir()
        member = {
            "slot": slot,
            "source_commit": c.SOURCES[slot],
            "seeds": list(c.SEEDS[slot]),
            "fixed_epochs": 8,
            "coordinator_sha256": str(slot) * 64,
            "root": str(root),
            "books_sha256": {str(seed): str(i + 1) * 64 for i, seed in enumerate(c.SEEDS[slot])},
            "admission_binding": str(tmp_path / f"admitted{slot}.json"),
            "terminal_owned_release": str(tmp_path / f"terminal{slot}.json"),
        }
        regpath, _ = write(
            tmp_path / f"reg{slot}.json",
            {
                "status": "frozen-before-formal-execution",
                "qualification_ledger_slot": slot,
                "source_commit": c.SOURCES[slot],
                "fixed_epochs": 8,
                "seeds": list(c.SEEDS[slot]),
            },
        )
        qpath, _ = write(
            tmp_path / f"Q{slot}.json",
            {
                "status": "frozen-before-formal-execution",
                "qualification_ledger_slot": slot,
                "source_commit": c.SOURCES[slot],
                "fixed_epochs": 8,
                "seeds": list(c.SEEDS[slot]),
                "books_sha256": member["books_sha256"],
                "helper_sha256": {f"own{slot}_posttraining.py": member["coordinator_sha256"]},
            },
        )
        member["registration"] = regpath
        member["qualification"] = qpath
        members.append(member)
    descriptor = {
        "schema": "prospective-own67-latency-final-cohort-v1",
        "status": "frozen-before-any-cohort-training-or-games",
        "slots": [6, 7],
        "latency_order": [6, 7],
        "admission7_deadline_epoch": c.ADMIT7,
        "prelatency_deadline_epoch": c.PRELAT,
        "latest_latency_start_epoch": c.LATEST,
        "hard_deadline_epoch": c.END,
        "conditional_release": (
            "terminal-INCOMPLETE-plus-all-owned-compute-ended;no-fabricated-ready-latency-finals"
        ),
        "helper_sha256": c.sha(HERE / "own67_cohort.py"),
        "members": members,
    }
    return descriptor


def admit(descriptor, slot):
    member = next(row for row in descriptor["members"] if row["slot"] == slot)
    c.register_member(
        descriptor, slot, member["registration"], member["qualification"], c.ADMIT7 - 1
    )
    return member


def ready(descriptor, slot, monkeypatch):
    member = admit(descriptor, slot)
    root = Path(member["root"])
    models = []
    for seed in c.SEEDS[slot]:
        path = root / f"model-{seed}"
        path.write_bytes(b"actual-test-model-" + str(seed).encode())
        models.append({"seed": seed, "candidate": str(path), "candidate_sha256": c.sha(path)})
        write(
            root / f"replay-{seed}-process-result.json",
            {
                "returncode": 0,
                "finished_epoch": c.ADMIT7,
                "deadline_epoch": c.ADMIT7 + 600,
            },
        )
    for phase in ("eligibility", "cuda-parity"):
        write(
            root / f"{phase}-process-result.json",
            {
                "returncode": 0,
                "finished_epoch": c.ADMIT7,
                "deadline_epoch": c.ADMIT7 + 120,
            },
        )
    write(
        root / "eligibility.json",
        {
            "fixed_epochs": 8,
            "qualification_ledger_slot": slot,
            "status": (
                "eligible-both-fixedSEARCH_ACTINGv2-for-preregistered-strength-only"
                if slot == 6
                else (
                    "eligible-both-fixedMETHOD7-native-v2-ledger-v3-for-preregistered-strength-only"
                )
            ),
            "seeds": [
                {
                    "seed": seed,
                    "epoch": 8,
                    "source_commit": c.SOURCES[slot],
                    "candidate_sha256": c.sha(root / f"model-{seed}"),
                }
                for seed in c.SEEDS[slot]
            ],
        },
    )
    monkeypatch.setattr(c.time, "time", lambda: c.PRELAT - 10)
    c.publish_ready(
        descriptor,
        member,
        {str(root / "eligibility.json"): c.sha(root / "eligibility.json")},
        models,
    )
    return member


def terminal(descriptor, slot, tmp_path, reason="never-admitted-before-fixed02:25"):
    member = next(row for row in descriptor["members"] if row["slot"] == slot)
    failure, failureSHA = write(
        tmp_path / "failure.json",
        {
            "status": "never-admitted-INCOMPLETE-fixed-deadline",
            "deadline_epoch": c.ADMIT7,
        },
    )
    before, bsha = write(
        tmp_path / "before.json",
        {
            "slot": slot,
            "source_commit": c.SOURCES[slot],
            "tracked_owned_pid_startticks": [{"pid": 99999999, "startticks": 1234}],
            "owned_process_group_ids": [99999999],
        },
    )
    after, asha = write(
        tmp_path / "after.json",
        {
            "slot": slot,
            "source_commit": c.SOURCES[slot],
            "remaining_owned_pid_startticks": [],
        },
    )
    value = {
        "schema": "own67-permanent-INCOMPLETE-owned-release-v1",
        "status": "INCOMPLETE-no-qualification-or-early-candidate-selection",
        "slot": slot,
        "source_commit": c.SOURCES[slot],
        "coordinator_sha256": member["coordinator_sha256"],
        "all_registered_owned_compute_terminated": True,
        "ready_latency_final_receipts_fabricated": False,
        "original_clocks_or_budgets_reset": False,
        "observed_epoch": c.ADMIT7 + 1,
        "reason": reason,
        "actual_failure": failure,
        "actual_failure_sha256": failureSHA,
        "before_inventory": before,
        "before_inventory_sha256": bsha,
        "after_inventory": after,
        "after_inventory_sha256": asha,
        "terminated_tracked_pid_startticks": [{"pid": 99999999, "startticks": 1234}],
    }
    return member, value


def latency(member):
    root = Path(member["root"])
    _, processSHA = write(
        root / "latency-process-result.json",
        {
            "returncode": 0,
            "finished_epoch": c.PRELAT + 10,
            "deadline_epoch": c.PRELAT + 180,
        },
    )
    _, latSHA = write(
        root / "latency.json",
        {
            "qualification_ledger_slot": member["slot"],
            "fixed_epochs": 8,
            "source_commit": member["source_commit"],
        },
    )
    return {
        "schema": "own67-real-latency-owner-completion-v1",
        "slot": member["slot"],
        "source_commit": member["source_commit"],
        "coordinator_sha256": member["coordinator_sha256"],
        "finished_epoch": c.PRELAT + 11,
        "process_receipt_sha256": processSHA,
        "latency_sha256": latSHA,
    }


def test_admission_once_exact_source_epoch_book_and_deadline(cohort):
    member = admit(cohort, 7)
    assert c.admission(member)["slot"] == 7
    with pytest.raises(FileExistsError):
        c.register_member(cohort, 7, member["registration"], member["qualification"], c.ADMIT7)
    Path(member["admission_binding"]).unlink()
    with pytest.raises(AssertionError):
        c.register_member(cohort, 7, member["registration"], member["qualification"], c.ADMIT7 + 1)


def test_real_ready_requires_all_native_proofs_and_exact_fixed8_models(cohort, monkeypatch):
    member = ready(cohort, 6, monkeypatch)
    value = c.read(Path(member["root"]) / "cohort67-ready.json")
    c.validate_ready(member, value)
    bad = copy.deepcopy(value)
    bad["models"][0]["candidate_sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        c.validate_ready(member, bad)
    bad = copy.deepcopy(value)
    bad["process_receipt_sha256"].pop("cuda-parity-process-result.json")
    with pytest.raises(AssertionError):
        c.validate_ready(member, bad)


def test_never_admitted7_can_release6_only_after_real_terminal_owned_proof(
    cohort, tmp_path, monkeypatch
):
    ready(cohort, 6, monkeypatch)
    member, value = terminal(cohort, 7, tmp_path)
    c.validate_terminal(member, value, now=c.PRELAT - 1, table={})
    write(member["terminal_owned_release"], value)
    assert c.wait_all_prepared(cohort, 6)[7][0] == "INCOMPLETE"
    assert not (Path(member["root"]) / "cohort67-ready.json").exists()
    assert not (Path(member["root"]) / "cohort67-latency-complete.json").exists()


def test_terminal_cannot_hide_live_owner_or_reset_clock(cohort, tmp_path):
    member, value = terminal(cohort, 7, tmp_path)
    table = {99999999: {"startticks": 1234, "state": "S", "pgid": 99999999}}
    with pytest.raises(AssertionError):
        c.validate_terminal(member, value, now=c.PRELAT - 1, table=table)
    with pytest.raises(AssertionError):
        c.validate_terminal(
            member,
            {**value, "original_clocks_or_budgets_reset": True},
            now=c.PRELAT - 1,
            table={},
        )
    with pytest.raises(AssertionError):
        c.validate_terminal(
            member,
            {**value, "observed_epoch": c.ADMIT7 - 1},
            now=c.PRELAT - 1,
            table={},
        )


def test_both_latency_order_and_final_release_without_gpu_or_arena_overlap(cohort, monkeypatch):
    first = ready(cohort, 6, monkeypatch)
    second = ready(cohort, 7, monkeypatch)
    prepared = c.wait_all_prepared(cohort, 6)
    digests = {
        str(peer["slot"]): c.sha(Path(peer["root"]) / "cohort67-ready.json")
        for peer in cohort["members"]
    }
    # Otherowner's ACK proves expensive hashing is already complete before thisownerlatency.
    write(
        Path(second["root"]) / "cohort67-prelat-validation-ack.json",
        {
            "schema": "own67-prelatency-artifact-validation-ack-v1",
            "slot": 7,
            "coordinator_sha256": second["coordinator_sha256"],
            "ready_or_terminal_sha256": digests,
        },
    )
    c.before_latency(cohort, first)
    assert prepared[6][0] == prepared[7][0] == "ready"
    write(Path(first["root"]) / "cohort67-latency-complete.json", latency(first))
    # The fixture ACK stands in for the concurrent second owner. Its actual call publishes once.
    (Path(second["root"]) / "cohort67-prelat-validation-ack.json").unlink()
    # Second sees genuine firstlat; it cannot fake or replace that receipt.
    c.before_latency(cohort, second)
    latency(second)
    monkeypatch.setattr(c.time, "time", lambda: c.PRELAT + 11)
    c.after_latency(cohort, second)
    assert (Path(second["root"]) / "cohort67-latency-complete.json").exists()


def test_changed_coordinator_or_receipt_source_rejected(cohort, monkeypatch):
    member = ready(cohort, 6, monkeypatch)
    value = latency(member)
    c.validate_latency(member, value)
    with pytest.raises(AssertionError):
        c.validate_latency(member, {**value, "source_commit": c.SOURCES[7]})
    with pytest.raises(AssertionError):
        c.validate_latency(member, {**value, "coordinator_sha256": "0" * 64})


def test_failed6_qualification_allows7_alone_without_fake6_latency(cohort, tmp_path, monkeypatch):
    second = ready(cohort, 7, monkeypatch)
    first, proof = terminal(
        cohort, 6, tmp_path, reason="actual-infrastructure-qualification-failure"
    )
    failed_path = Path(proof["actual_failure"])
    write(
        failed_path,
        {"status": "failed-or-incomplete-preserved-no-retry", "deadline_epoch": 1791164847.632503},
    )
    proof["actual_failure_sha256"] = c.sha(failed_path)
    write(first["terminal_owned_release"], proof)
    c.before_latency(cohort, second)
    latency(second)
    monkeypatch.setattr(c.time, "time", lambda: c.PRELAT + 11)
    c.after_latency(cohort, second)
    assert not (Path(first["root"]) / "cohort67-ready.json").exists()
    assert not (Path(first["root"]) / "cohort67-latency-complete.json").exists()


def test_missing_peer_is_incomplete_at_fixed_preparation_ceiling(cohort, monkeypatch):
    ready(cohort, 7, monkeypatch)
    monkeypatch.setattr(c.time, "time", lambda: c.PRELAT)
    with pytest.raises(TimeoutError):
        c.wait_all_prepared(cohort, 7)


def test_finals_cannot_release_before_other_real_latency(cohort, monkeypatch):
    first = ready(cohort, 6, monkeypatch)
    ready(cohort, 7, monkeypatch)
    latency(first)
    # Own real completed latency is valid; a missing peer never becomes success at END.
    monkeypatch.setattr(c.time, "time", lambda: c.END - 1)
    with pytest.raises(TimeoutError):
        c.after_latency(cohort, first)
