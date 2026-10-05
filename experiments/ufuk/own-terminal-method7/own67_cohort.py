"""Prospective6/7 latency/final barrier; terminal failures never masquerade as readiness."""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

END = 1791180000
ADMIT7 = 1791167100
PRELAT = 1791173460
LATEST = 1791174000
SOURCES = {
    6: "428a30f5e1658f3cf159844db547ff0147ade5a9",
    7: "c022bc1605b44c3089439da5c6efb7bd4db4ff81",
}
SEEDS = {6: (20261625, 20261626), 7: (20261725, 20261726)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix=".cohort-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def bound(path, digest):
    assert sha(path) == digest
    return json.loads(Path(path).read_text())


def bind(config, slot, coordinator_sha256, sha_fn=sha):
    info = config["cohort67"]
    path = Path(info["manifest"])
    assert sha_fn(path) == info["manifest_sha256"]
    descriptor = json.loads(path.read_text())
    assert descriptor["schema"] == "prospective-own67-latency-final-cohort-v1"
    assert descriptor["status"] == "frozen-before-any-cohort-training-or-games"
    assert descriptor["slots"] == [6, 7] and descriptor["latency_order"] == [6, 7]
    assert (
        descriptor["admission7_deadline_epoch"] == ADMIT7
        and descriptor["prelatency_deadline_epoch"] == PRELAT
    )
    assert (
        descriptor["latest_latency_start_epoch"] == LATEST
        and descriptor["hard_deadline_epoch"] == END
    )
    assert (
        descriptor["conditional_release"]
        == "terminal-INCOMPLETE-plus-all-owned-compute-ended;no-fabricated-ready-latency-finals"
    )
    assert [row["slot"] for row in descriptor["members"]] == [6, 7]
    for row in descriptor["members"]:
        assert row["source_commit"] == SOURCES[row["slot"]]
        assert row["seeds"] == list(SEEDS[row["slot"]]) and row["fixed_epochs"] == 8
        assert set(row["books_sha256"]) == set(map(str, SEEDS[row["slot"]]))
        assert len(set(row["books_sha256"].values())) == 2
    member = next(row for row in descriptor["members"] if row["slot"] == slot)
    assert member["source_commit"] == config["source_commit"] == SOURCES[slot]
    assert member["seeds"] == list(SEEDS[slot])
    assert member["fixed_epochs"] == 8 and member["coordinator_sha256"] == coordinator_sha256
    assert member["root"] == config["root"]
    q = bound(config["qualification_config"], config["qualification_config_sha256"])
    assert (
        q["fixed_epochs"] == 8
        and q["qualification_ledger_slot"] == slot
        and q["source_commit"] == SOURCES[slot]
    )
    assert q["books_sha256"] == member["books_sha256"]
    assert descriptor["helper_sha256"] == sha_fn(__file__)
    return descriptor, member


def register_member(descriptor, slot, registration_path, qualification_path, observed=None):
    observed = time.time() if observed is None else observed
    member = next(row for row in descriptor["members"] if row["slot"] == slot)
    registration = json.loads(Path(registration_path).read_text())
    q = json.loads(Path(qualification_path).read_text())
    assert registration["status"] == q["status"] == "frozen-before-formal-execution"
    assert registration["qualification_ledger_slot"] == q["qualification_ledger_slot"] == slot
    assert registration["source_commit"] == q["source_commit"] == SOURCES[slot]
    assert registration["fixed_epochs"] == q["fixed_epochs"] == 8
    assert registration["seeds"] == q["seeds"] == list(SEEDS[slot])
    assert q["books_sha256"] == member["books_sha256"]
    assert q["helper_sha256"][f"own{slot}_posttraining.py"] == member["coordinator_sha256"]
    if slot == 7:
        assert observed <= ADMIT7
    receipt = {
        "schema": "own67-qualified-member-admission-v1",
        "status": "admitted-before-compute",
        "slot": slot,
        "source_commit": SOURCES[slot],
        "fixed_epochs": 8,
        "observed_epoch": observed,
        "registration": str(Path(registration_path).resolve()),
        "registration_sha256": sha(registration_path),
        "qualification_config": str(Path(qualification_path).resolve()),
        "qualification_config_sha256": sha(qualification_path),
        "coordinator_sha256": member["coordinator_sha256"],
    }
    Path(member["admission_binding"]).parent.mkdir(parents=True, exist_ok=True)
    publish(member["admission_binding"], receipt)
    return receipt


def admission(member):
    receipt = read(member["admission_binding"])
    if receipt is None:
        return None
    assert (
        receipt["schema"] == "own67-qualified-member-admission-v1"
        and receipt["status"] == "admitted-before-compute"
    )
    slot = member["slot"]
    assert receipt["slot"] == slot and receipt["source_commit"] == SOURCES[slot]
    assert (
        receipt["fixed_epochs"] == 8
        and receipt["coordinator_sha256"] == member["coordinator_sha256"]
    )
    if slot == 7:
        assert receipt["observed_epoch"] <= ADMIT7
    registration = bound(receipt["registration"], receipt["registration_sha256"])
    q = bound(receipt["qualification_config"], receipt["qualification_config_sha256"])
    assert registration["fixed_epochs"] == q["fixed_epochs"] == 8
    assert registration["source_commit"] == q["source_commit"] == SOURCES[slot]
    assert q["books_sha256"] == member["books_sha256"]
    return receipt


def verify_process(path, digest):
    result = bound(path, digest)
    assert type(result["returncode"]) is int and result["returncode"] == 0
    assert result["finished_epoch"] <= result["deadline_epoch"] < END
    return result


def validate_ready(member, receipt):
    admitted = admission(member)
    assert admitted is not None
    assert (
        receipt["schema"] == "own67-prelatency-owner-readiness-v1"
        and receipt["slot"] == member["slot"]
    )
    assert receipt["source_commit"] == SOURCES[member["slot"]] and receipt["fixed_epochs"] == 8
    assert receipt["coordinator_sha256"] == member["coordinator_sha256"]
    assert receipt["admission_binding_sha256"] == sha(member["admission_binding"])
    assert receipt["finished_epoch"] <= PRELAT
    root = Path(member["root"])
    required = {
        "eligibility-process-result.json",
        "cuda-parity-process-result.json",
    } | {f"replay-{seed}-process-result.json" for seed in SEEDS[member["slot"]]}
    assert set(receipt["process_receipt_sha256"]) == required
    for name, digest in receipt["process_receipt_sha256"].items():
        verify_process(root / name, digest)
    artifacts = receipt["all_prevalidated_artifact_sha256"]
    assert artifacts
    for path, digest in artifacts.items():
        assert sha(path) == digest
    models = receipt["models"]
    assert [row["seed"] for row in models] == list(SEEDS[member["slot"]])
    for row in models:
        assert sha(row["candidate"]) == row["candidate_sha256"]
    eligible = bound(root / "eligibility.json", receipt["eligibility_sha256"])
    assert eligible["fixed_epochs"] == 8 and eligible["qualification_ledger_slot"] == member["slot"]
    expected_status = {
        6: "eligible-both-fixedSEARCH_ACTINGv2-for-preregistered-strength-only",
        7: "eligible-both-fixedMETHOD7-native-v2-ledger-v3-for-preregistered-strength-only",
    }
    assert eligible["status"] == expected_status[member["slot"]]
    assert [row["seed"] for row in eligible["seeds"]] == list(SEEDS[member["slot"]])
    assert [row["candidate_sha256"] for row in eligible["seeds"]] == [
        row["candidate_sha256"] for row in models
    ]
    assert all(
        row["epoch"] == 8 and row["source_commit"] == SOURCES[member["slot"]]
        for row in eligible["seeds"]
    )
    return receipt


def publish_ready(descriptor, member, artifacts, models):
    root = Path(member["root"])
    required = [
        "eligibility-process-result.json",
        "cuda-parity-process-result.json",
    ] + [f"replay-{seed}-process-result.json" for seed in SEEDS[member["slot"]]]
    receipt = {
        "schema": "own67-prelatency-owner-readiness-v1",
        "slot": member["slot"],
        "source_commit": SOURCES[member["slot"]],
        "fixed_epochs": 8,
        "coordinator_sha256": member["coordinator_sha256"],
        "finished_epoch": time.time(),
        "admission_binding_sha256": sha(member["admission_binding"]),
        "process_receipt_sha256": {name: sha(root / name) for name in required},
        "all_prevalidated_artifact_sha256": artifacts,
        "models": models,
        "eligibility_sha256": sha(root / "eligibility.json"),
    }
    validate_ready(member, receipt)
    publish(root / "cohort67-ready.json", receipt)


def proc_table():
    table = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            table[int(path.parent.name)] = {
                "state": fields[0],
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "startticks": int(fields[19]),
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return table


def validate_terminal(member, terminal, now=None, table=None):
    now = time.time() if now is None else now
    assert terminal["schema"] == "own67-permanent-INCOMPLETE-owned-release-v1"
    assert terminal["status"] == "INCOMPLETE-no-qualification-or-early-candidate-selection"
    assert (
        terminal["slot"] == member["slot"] and terminal["source_commit"] == SOURCES[member["slot"]]
    )
    assert terminal["coordinator_sha256"] == member["coordinator_sha256"]
    assert terminal["all_registered_owned_compute_terminated"] is True
    assert terminal["ready_latency_final_receipts_fabricated"] is False
    assert terminal["original_clocks_or_budgets_reset"] is False
    assert terminal["observed_epoch"] <= now < END
    failure = bound(terminal["actual_failure"], terminal["actual_failure_sha256"])
    if terminal["reason"] == "never-admitted-before-fixed02:25":
        assert (
            member["slot"] == 7
            and terminal["observed_epoch"] >= ADMIT7
            and admission(member) is None
        )
        assert (
            failure["status"] == "never-admitted-INCOMPLETE-fixed-deadline"
            and failure["deadline_epoch"] == ADMIT7
        )
    else:
        assert terminal["reason"] in (
            "actual-infrastructure-qualification-failure",
            "actual-owned-formal-phase-failure",
        )
        assert failure["status"] in (
            "failed",
            "failed-preserved",
            "failed-preserved-audit",
            "failed-or-incomplete-preserved-no-retry",
            "failed-preserved-mutation",
            "failed-preserved-freshprocess",
            "failed-or-incomplete",
        )
    before = bound(terminal["before_inventory"], terminal["before_inventory_sha256"])
    after = bound(terminal["after_inventory"], terminal["after_inventory_sha256"])
    assert before["slot"] == after["slot"] == member["slot"]
    assert before["source_commit"] == after["source_commit"] == SOURCES[member["slot"]]
    assert after["remaining_owned_pid_startticks"] == []
    tracked = before["tracked_owned_pid_startticks"]
    assert terminal["terminated_tracked_pid_startticks"] == tracked
    actual = proc_table() if table is None else table
    assert not any(
        actual.get(row["pid"], {}).get("startticks") == row["startticks"]
        and actual[row["pid"]]["state"] not in ("Z", "X")
        for row in tracked
    )
    assert not any(
        row["state"] not in ("Z", "X") and row["pgid"] in set(before["owned_process_group_ids"])
        for row in actual.values()
    )
    return terminal


def member_state(member, now=None):
    now = time.time() if now is None else now
    # A terminal receipt is not optional stopping; the actual permanent failure stays archived.
    terminal = read(member["terminal_owned_release"])
    if terminal is not None:
        return "INCOMPLETE", validate_terminal(member, terminal, now)
    ready = read(Path(member["root"]) / "cohort67-ready.json")
    if ready is not None:
        return "ready", validate_ready(member, ready)
    return "pending", None


def wait_all_prepared(descriptor, ownslot):
    prepared = {}
    while True:
        for row in descriptor["members"]:
            if row["slot"] not in prepared:
                state = member_state(row)
                if state[0] in ("ready", "INCOMPLETE"):
                    prepared[row["slot"]] = state
        if ownslot in prepared and prepared[ownslot][0] == "INCOMPLETE":
            raise RuntimeError("Own family permanently INCOMPLETE")
        if len(prepared) == 2:
            return prepared
        if time.time() >= PRELAT:
            raise TimeoutError("Fixed cohort04:11 prelatency ceiling exhausted")
        time.sleep(0.5)


def validate_latency(member, receipt):
    root = Path(member["root"])
    assert receipt["schema"] == "own67-real-latency-owner-completion-v1"
    assert receipt["slot"] == member["slot"] and receipt["source_commit"] == SOURCES[member["slot"]]
    assert receipt["coordinator_sha256"] == member["coordinator_sha256"]
    assert receipt["process_receipt_sha256"] == sha(root / "latency-process-result.json")
    result = verify_process(root / "latency-process-result.json", receipt["process_receipt_sha256"])
    assert result["finished_epoch"] <= receipt["finished_epoch"] < END
    assert receipt["latency_sha256"] == sha(root / "latency.json")
    latency = bound(root / "latency.json", receipt["latency_sha256"])
    assert latency["source_commit"] == SOURCES[member["slot"]]
    assert latency["qualification_ledger_slot"] == member["slot"]
    assert latency["fixed_epochs"] == 8
    return receipt


def before_latency(descriptor, member):
    prepared = wait_all_prepared(descriptor, member["slot"])
    # All expensive immutable artifact checks finish on BOTH owners before first latency.
    root = Path(member["root"])
    ack = {
        "schema": "own67-prelatency-artifact-validation-ack-v1",
        "slot": member["slot"],
        "coordinator_sha256": member["coordinator_sha256"],
        "ready_or_terminal_sha256": {
            str(peer["slot"]): sha(
                Path(peer["root"]) / "cohort67-ready.json"
                if prepared[peer["slot"]][0] == "ready"
                else peer["terminal_owned_release"]
            )
            for peer in descriptor["members"]
        },
    }
    publish(root / "cohort67-prelat-validation-ack.json", ack)
    while True:
        acknowledged = True
        for peer in descriptor["members"]:
            if prepared[peer["slot"]][0] == "INCOMPLETE":
                continue
            value = read(Path(peer["root"]) / "cohort67-prelat-validation-ack.json")
            if value is None:
                acknowledged = False
                continue
            assert value["schema"] == ack["schema"] and value["slot"] == peer["slot"]
            assert value["coordinator_sha256"] == peer["coordinator_sha256"]
            assert value["ready_or_terminal_sha256"] == ack["ready_or_terminal_sha256"]
        if acknowledged:
            break
        if time.time() >= PRELAT:
            raise TimeoutError("Both artifact-validation acknowledgements required04:11")
        time.sleep(0.5)
    if member["slot"] == 7:
        peer = next(row for row in descriptor["members"] if row["slot"] == 6)
        while prepared[6][0] != "INCOMPLETE":
            terminal = read(peer["terminal_owned_release"])
            if terminal is not None:
                validate_terminal(peer, terminal)
                break
            receipt = read(Path(peer["root"]) / "cohort67-latency-complete.json")
            if receipt is not None:
                validate_latency(peer, receipt)
                break
            if time.time() >= LATEST:
                raise TimeoutError("6latency completion before7 required")
            time.sleep(0.5)
    assert time.time() < LATEST


def after_latency(descriptor, member):
    root = Path(member["root"])
    receipt = {
        "schema": "own67-real-latency-owner-completion-v1",
        "slot": member["slot"],
        "source_commit": SOURCES[member["slot"]],
        "coordinator_sha256": member["coordinator_sha256"],
        "finished_epoch": time.time(),
        "process_receipt_sha256": sha(root / "latency-process-result.json"),
        "latency_sha256": sha(root / "latency.json"),
    }
    validate_latency(member, receipt)
    publish(root / "cohort67-latency-complete.json", receipt)
    while True:
        all_finished = True
        for peer in descriptor["members"]:
            terminal = read(peer["terminal_owned_release"])
            if terminal is not None:
                validate_terminal(peer, terminal)
                continue
            result = read(Path(peer["root"]) / "cohort67-latency-complete.json")
            if result is None:
                all_finished = False
                continue
            validate_latency(peer, result)
        if all_finished:
            return
        if time.time() >= END - 1:
            raise TimeoutError(
                "Both genuine latencies or terminal failures required before final games"
            )
        time.sleep(0.5)
