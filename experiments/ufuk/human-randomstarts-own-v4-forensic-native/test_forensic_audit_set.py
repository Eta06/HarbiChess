import json
from pathlib import Path

import pytest

import forensic_audit_set as auditset


def _ref(path: Path) -> dict:
    return {"path": str(path), "sha256": auditset.sha(path)}


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _fixture(tmp_path: Path) -> dict:
    audits = []
    runtime = tmp_path / "audit_runtime.py"
    runtime.write_text("# runtime\n")
    validator = Path(auditset.__file__).resolve()
    for seed in auditset.SEEDS:
        rows = [f"root:{i}" for i in range(1024)]
        periodic = [rows[i] for i in auditset.OFFSETS]
        reg_path = _write(
            tmp_path / f"registration-{seed}.json",
            {
                "schema": "human-randomstarts-own-collection-registration-v2",
                "status": "registered",
                "seed": seed,
                "generation": 1,
                "parent_admission_result": {"path": "parent", "sha256": "a" * 64},
            },
        )
        events_path = tmp_path / f"events-{seed}.jsonl"
        events_path.write_text('{"type":"fixture"}\n')
        receipt_path = _write(
            tmp_path / f"receipt-{seed}.json",
            {
                "schema": "human-randomstarts-own-collection-receipt-v2",
                "status": "PASS-exact-row-budget",
                "seed": seed,
                "generation": 1,
                "train_rows": 1024,
                "teacher_labels_used": False,
                "registration_sha256": "f" * 64,
                "events_sha256": _ref(events_path)["sha256"],
                "training_row_ids": rows,
                "periodic_independent_search_rows": periodic,
            },
        )
        files = {
            "registration": _ref(reg_path),
            "receipt": _ref(receipt_path),
            "events": _ref(events_path),
        }
        view_path = _write(
            tmp_path / f"view-{seed}.json",
            {
                "schema": "human-randomstarts-own-forensic-receipt-view-v3",
                "status": "PASS-source-shadow-only-audit-required",
                "seed": seed,
                "generation": 1,
                "raw_receipt": files["receipt"],
                "original_registration": files["registration"],
                "raw_receipt_registration_sha256": "f" * 64,
                "verified_original_registration_sha256": files["registration"]["sha256"],
            },
        )
        files["forensic_view"] = _ref(view_path)
        clock = {
            "schema": auditset.CLOCK_SCHEMA,
            "seed": seed,
            "helper_sha256": _ref(runtime)["sha256"],
            **files,
            "registration_sha256": files["registration"]["sha256"],
            "receipt_sha256": files["receipt"]["sha256"],
            "events_sha256": files["events"]["sha256"],
            "forensic_view_sha256": files["forensic_view"]["sha256"],
            "first": 100.0,
            "deadline": 200.0,
            "operator_end_epoch": 250.0,
        }
        clock_path = _write(tmp_path / f"clock-{seed}.json", clock)
        result = {
            "schema": auditset.RESULT_SCHEMA,
            "status": auditset.STATUS,
            "seed": seed,
            "generation": 1,
            "parent_admission_result": {"path": "parent", "sha256": "a" * 64},
            "helper_sha256": _ref(runtime)["sha256"],
            "clock_sha256": _ref(clock_path)["sha256"],
            "registration_sha256": files["registration"]["sha256"],
            "receipt_sha256": files["receipt"]["sha256"],
            "events_sha256": files["events"]["sha256"],
            "forensic_view_sha256": files["forensic_view"]["sha256"],
            "raw_receipt_registration_sha256": "f" * 64,
            "verified_original_registration_sha256": files["registration"]["sha256"],
            "receipt_view_is_not_a_rewritten_v2_receipt": True,
            "packets": [{"row_id": row} for row in periodic],
            "new_training_rows": 0,
            "new_games": 0,
            "optimizer_updates": 0,
            "first": 100.0,
            "deadline": 200.0,
            "finished": 150.0,
            "raw_receipt_path": files["receipt"]["path"],
            "forensic_view_path": files["forensic_view"]["path"],
        }
        result_path = _write(tmp_path / f"result-{seed}.json", result)
        audits.append(
            {
                "seed": seed,
                **files,
                "result": _ref(result_path),
                "clock": _ref(clock_path),
                "raw_wrong_registration_sha256": "f" * 64,
                "operator_end_epoch": 250.0,
                "training_row_ids": rows,
                "periodic_search_rows": periodic,
            }
        )
    return {
        "schema": auditset.SCHEMA,
        "status": "PASS-both-forensic-six-packet-audits",
        "helper": _ref(runtime),
        "validator": _ref(validator),
        "audits": audits,
    }


def test_both_seed_forensic_audit_set_is_accepted(tmp_path):
    normalized = auditset.validate(_fixture(tmp_path))
    assert normalized["schema"] == auditset.SCHEMA
    assert len(normalized["audits"]) == 2


def test_forensic_audit_set_rejects_resealed_wrong_packet_ordinal(tmp_path):
    manifest = _fixture(tmp_path)
    item = manifest["audits"][0]
    result = json.loads(Path(item["result"]["path"]).read_bytes())
    result["packets"][0]["row_id"] = "different"
    Path(item["result"]["path"]).write_text(json.dumps(result, sort_keys=True))
    item["result"] = _ref(Path(item["result"]["path"]))
    with pytest.raises(ValueError, match="six exact chronological packet rows"):
        auditset.validate(manifest)


def test_forensic_audit_set_rejects_mutated_raw_wrong_sha(tmp_path):
    manifest = _fixture(tmp_path)
    item = manifest["audits"][0]
    raw = json.loads(Path(item["receipt"]["path"]).read_bytes())
    raw["registration_sha256"] = "0" * 64
    Path(item["receipt"]["path"]).write_text(json.dumps(raw, sort_keys=True))
    item["receipt"] = _ref(Path(item["receipt"]["path"]))
    with pytest.raises(ValueError, match="forensic six-audit result/clock/input binding"):
        auditset.validate(manifest)
