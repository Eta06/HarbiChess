"""Metadata-only actual-input wiring and closed-phase rejection tests; no child execution."""

import json
from pathlib import Path

import execute
import pytest

BASE = Path("/workspace/work/harbichess/continuation-20261007")


@pytest.mark.parametrize("seed", execute.SEEDS)
def test_actual_common_input_bindings_keep_old_seal_and_named_schema(tmp_path, seed):
    data = execute.inputs(BASE, seed, tmp_path)
    common = Path(f"/dev/shm/harbichess-human-randomstarts-forensic-data-v4/{seed}")
    provenance = execute.read(common / "provenance.json")
    assert (
        execute.read(data["conversion"]["common_conversion_seal"]["path"]) == provenance["inputs"]
    )
    assert data["conversion"]["common_dataset"] == execute.ref(common / "dataset.json")
    assert data["conversion"]["common_result"] == execute.ref(common / "result.json")
    assert data["conversion"]["forensic_inventories"][0]["sha256"] == execute.FORENSIC
    assert data["search_helper"]["sha256"] == (
        "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670"
    )
    assert "teacher" not in data
    assert not execute.output("proof", seed).is_relative_to(tmp_path)


def test_incomplete_phase_is_not_accepted_and_changed_artifact_rejected(tmp_path):
    path = execute.records(tmp_path, "proof", execute.SEEDS[0])
    path.mkdir()
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    result = path / "result.json"
    value = dict(
        status="FAILED-preserved",
        first=1,
        finished=2,
        deadline=3,
        artifacts={"contract": execute.ref(artifact)},
    )
    result.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="did not PASS"):
        execute.completed(tmp_path, "proof", execute.SEEDS[0])
    value["status"] = "PASS-actual-antisymmetric-proof-not-strength"
    result.write_text(json.dumps(value))
    assert execute.completed(tmp_path, "proof", execute.SEEDS[0])["finished"] == 2
    artifact.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="artifact changed"):
        execute.completed(tmp_path, "proof", execute.SEEDS[0])


def test_protocol_does_not_inherit_teacher_endpoint_or_old_expired_clock(tmp_path, monkeypatch):
    fixture = tmp_path / "newzero.json"
    fixture.write_text("{}")
    binding = {name: execute.ref(fixture) for name in ["candidate", "native", "contract", "result"]}
    monkeypatch.setattr(execute, "zero_binding", lambda *args: binding)
    monkeypatch.setattr(execute, "completed", lambda *args: {})
    monkeypatch.setattr(execute, "output", lambda *args: tmp_path)
    (tmp_path / "dataset.json").write_text("{}")
    first = 1791400000.0
    q = execute.protocol(BASE, execute.stage(BASE), "zero-profile", first, 1791448916.685839)
    assert q["original_first_epoch"] == first
    assert q["original_deadline_epoch"] == first + 7200
    assert q["children"] == {}
    assert set(q["models"][str(execute.SEEDS[0])]) == {"parent", "e8"}
    assert not {"teachers", "nnue_helpers", "collection_audit_set"} & q.keys()
    assert q["scope"]["next_generation_supported"] is False
    assert q["models"][str(execute.SEEDS[0])]["e8"] == q["models"][str(execute.SEEDS[1])]["e8"]
