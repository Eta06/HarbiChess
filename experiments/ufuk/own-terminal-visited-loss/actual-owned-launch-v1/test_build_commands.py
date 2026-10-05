import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("planner", HERE / "build_commands.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_placeholder_inputs_cannot_become_launch_argv():
    value = json.loads((HERE / "command-plan-TEMPLATE.json").read_text())
    with pytest.raises(AssertionError):
        p.factory(value)
    with pytest.raises(AssertionError):
        p.owners(value)
    with pytest.raises(AssertionError):
        p.launches(value)


def test_owner_paths_bind_actual_post_field_names_and_no_fake_success():
    value = json.loads((HERE / "owner-paths-TEMPLATE.json").read_text())
    for row in value["seeds"].values():
        assert {
            "run",
            "training_result",
            "audit_controller_result",
            "full_audit",
            "book",
            "baseline_root",
            "baseline_deadline_epoch",
            "final_root",
            "original_training_started_epoch",
        } <= row.keys()
    assert value["posttraining"]["expected_A100_runtime"]["torch_version"] == "2.11.0+cu130"
    assert "PENDING" in value["status"]


def test_book_conversion_requires_no_engine_or_model_queries():
    value = json.loads((HERE / "books-provenance.json").read_text())
    assert value["qualification_ledger_slot"] == 8
    assert value["unique_source_records_across_both"] == 96
    assert value["root_overlap"] == value["source_overlap"] == 0
    assert set(value["selection_to_training_seed"].values()) == {20261825, 20261826}
    assert value["model_or_engine_queries"] == value["games_played"] == 0
    assert len(set(value["book_sha256"].values())) == 2


def test_staging_checks_all_existing_files_before_any_write(tmp_path):
    spec = importlib.util.spec_from_file_location("stage", HERE / "prepare_immutable_stage.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    source = tmp_path / "source"
    source.write_bytes(b"expected")
    existing = tmp_path / "existing"
    existing.write_bytes(b"historical-conflict")
    missing = tmp_path / "must-stay-missing"
    code = stage.render({str(missing): source, str(existing): source})
    with pytest.raises(AssertionError):
        exec(compile(code, "test-local-staging", "exec"), {})
    assert not missing.exists()
    assert existing.read_bytes() == b"historical-conflict"


def test_staging_same_bytes_can_be_verified_without_overwrite(tmp_path):
    spec = importlib.util.spec_from_file_location("stage", HERE / "prepare_immutable_stage.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    source = tmp_path / "source"
    source.write_bytes(b"expected")
    destination = tmp_path / "destination"
    code = stage.render({str(destination): source})
    exec(compile(code, "test-local-staging", "exec"), {})
    first_inode = destination.stat().st_ino
    exec(compile(code, "test-local-staging", "exec"), {})
    assert destination.stat().st_ino == first_inode
    assert destination.read_bytes() == b"expected"
