import json
from pathlib import Path

import pytest

import forensic_receipt as forensic


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return path


def _fixture(tmp_path: Path):
    base = tmp_path / "continuation"
    seed = 20262905
    producer = base / "producer"
    producer.mkdir(parents=True)
    helper = producer / "parent_bridge.py"
    helper.write_text("# frozen helper\n")
    run = producer / "run_collection.py"
    run.write_text(
        "def execute(path):\n"
        "    reg = {}\n"
        "    for path, digest in reg['generation_helper_sha256'].items():\n"
        "        pass\n"
        "    receipt = {'registration_sha256': sha(path)}\n"
    )
    source_sha = forensic.sha(run)
    helper_sha = forensic.sha(helper)
    input_pin = base / "command.py"
    input_pin.write_text("# command input\n")
    parent = {
        "path": "/dev/shm/parent.pt",
        "sha256": "a" * 64,
        "contract_sha256": "0" * 64,
    }
    admission = {"path": "/work/admission.json", "sha256": "b" * 64}
    protected = {"path": "/dev/shm/protected.bin", "sha256": "d" * 64}
    search = {"path": "/work/search.py", "sha256": "e" * 64}
    bank = {"path": "/dev/shm/bank.json", "sha256": "f" * 64}
    helpers = {"parent": "same"}
    output_dir = base / "actual-output"
    output_dir.mkdir()
    events = output_dir / "events.jsonl"
    events.write_text("")
    source_map = {"run_collection.py": source_sha}
    helper_map = {str(helper): helper_sha}
    search_config = {"nodes": 8192, "qdepth": 2, "max_depth": 8}
    selection_path = "/dev/shm/selection.json"
    selection_sha = "c" * 64
    reg = {
        "schema": forensic.REG_SCHEMA,
        "status": "registered",
        "seed": seed,
        "generation": 1,
        "cpu_core": 1,
        "operator_end_epoch": 2000.0,
        "original_first_epoch": 1000.0,
        "original_deadline_epoch": 1500.0,
        "search": search_config,
        "core_repo": "/work/core",
        "core_commit": "1234567890abcdef",
        "parent_candidate": parent,
        "parent_admission_result": admission,
        "parent_admission_seal": {"path": "/work/seal.json", "sha256": "1" * 64},
        "parent_helpers": helpers,
        "protected_aliases": protected,
        "search_helper": search,
        "procedural_bank_receipt": bank,
        "root_pool": {
            "path": "/dev/shm/pool.json",
            "sha256": "c" * 64,
            "selection_path": selection_path,
            "selection_sha256": selection_sha,
        },
        "output_path": str(base / "actual-output"),
        "producer_source_sha256": source_map,
        "generation_helper_sha256": helper_map,
    }
    reg_dir = base / f"human-random-collect-actual-{seed}"
    reg_path = _write(reg_dir / "collection-registration.json", reg)
    raw = {
        "schema": forensic.RAW_SCHEMA,
        "status": "PASS-exact-row-budget",
        "seed": seed,
        "generation": 1,
        "train_rows": 1024,
        "teacher_labels_used": False,
        "registration_sha256": helper_sha,
        "parent_candidate_sha256": parent["sha256"],
        "parent_contract_sha256": parent["contract_sha256"],
        "parent_helpers": helpers,
        "parent_admission_result": admission,
        "parent_admission_seal": reg["parent_admission_seal"],
        "search_helper_sha256": search["sha256"],
        "root_pool_sha256": reg["root_pool"]["sha256"],
        "protected_aliases_sha256": protected["sha256"],
        "procedural_selection_path": selection_path,
        "procedural_selection_sha256": selection_sha,
        "procedural_bank_receipt": bank,
        "source_selection_sha256": selection_sha,
        "search": search_config,
        "original_first_epoch": 1000.0,
        "original_deadline_epoch": 1500.0,
        "operator_end_epoch": 2000.0,
        "producer_source_sha256": source_map,
        "generation_helper_sha256": helper_map,
        "target_source": "own-search; own WDL only for completed episodes",
        "exposure_definition": (
            "all path boards plus actual static evaluator inputs; "
            "per-row evaluator aliases in sorted-unique sidecars"
        ),
        "events_bytes": events.stat().st_size,
        "events_sha256": forensic.sha(events),
        "finished_epoch": 1200.0,
        "all_actor_rows": 1024,
        "starts_considered": 1,
        "unique_exposure_aliases": 1,
        "exposed_board_alias_count": 1,
    }
    receipt_path = _write(base / "receipt.json", raw)
    seal = {
        "schema": "human-randomstarts-own-collection-build-seal-v2",
        "status": "registered",
        "generation": 1,
        "cpu_core": 1,
        "first": 1000.0,
        "deadline": 1500.0,
        "operator_end_epoch": 2000.0,
        "core_repo": "/work/core",
        "core_commit": "1234567890abcdef",
        "parent_admission_result": admission,
        "parent_admission_seal": reg["parent_admission_seal"],
        "parent_helpers": helpers,
        "protected_aliases": protected,
        "search_helper": search,
        "procedural_bank_receipt": bank,
        "root_pool_output": "/dev/shm/pool.json",
    }
    _write(reg_dir / "collection-build-seal.json", seal)
    controller_dir = base / f"human-random-collect-{seed}-controller"
    controller_path = controller_dir / "controller-registration.json"
    controller = {
        "schema": "UFUK-DEVAM-bounded-cpu-phase-v1",
        "phase": "g0 procedural4096 seed05",
        "cpu_core": 1,
        "checkout": "/work/core",
        "operator_end_epoch": 2000.0,
        "first_epoch": 900.0,
        "deadline_epoch": 1600.0,
        "controller_sha256": "9" * 64,
        "input_pins": {str(input_pin): forensic.sha(input_pin)},
    }
    _write(controller_path, controller)
    _write(
        controller_dir / "owner-process.json",
        {
            "registration": str(controller_path),
            "first_epoch": 900.0,
            "deadline_epoch": 1600.0,
            "operator_end_epoch": 2000.0,
        },
    )
    return reg_path, receipt_path, producer, helper_sha


def test_shadow_derivation_preserves_raw_field_and_pins_pre_actor_context(tmp_path):
    reg_path, receipt_path, producer, helper_sha = _fixture(tmp_path)
    view = forensic.derive_view(reg_path, receipt_path, producer)
    assert view["schema"] == forensic.SCHEMA
    assert view["status"] == "PASS-source-shadow-only-audit-required"
    assert view["raw_receipt_registration_sha256"] == helper_sha
    assert view["verified_original_registration_sha256"] == forensic.sha(reg_path)
    assert view["raw_receipt_bytes_preserved"] is True
    assert view["receipt_view_is_not_a_rewritten_v2_receipt"] is True
    assert view["root_launch_evidence"]["build_seal"]["sha256"]
    assert len(view["root_launch_evidence"]["controller_input_pins"]) == 1


def test_shadow_derivation_rejects_any_other_wrong_receipt_binding(tmp_path):
    reg_path, receipt_path, producer, _ = _fixture(tmp_path)
    raw = json.loads(receipt_path.read_bytes())
    raw["registration_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="raw incorrect field"):
        forensic.derive_view(reg_path, receipt_path, producer)


def test_source_detector_requires_path_loop_then_receipt_hash(tmp_path):
    source = tmp_path / "run.py"
    source.write_text(
        "def execute(path):\n"
        "    for path, digest in reg['generation_helper_sha256'].items():\n"
        "        pass\n"
        "    receipt = {'registration_sha256': sha(path)}\n"
    )
    assert forensic.source_has_path_shadow_bug(source)
    source.write_text(
        "def execute(path):\n"
        "    for path, digest in reg['generation_helper_sha256'].items():\n"
        "        pass\n"
        "    receipt = {'registration_sha256': sha(reg_path)}\n"
    )
    assert not forensic.source_has_path_shadow_bug(source)
