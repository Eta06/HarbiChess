"""Pure metadata projection tests; never import or execute the actual trainer."""

import hashlib
import json

import pytest
from adapter import EXECUTOR, PUBLIC, project, sha


def fixture(tmp_path):
    files = {}
    for name in ("prove", "train", "native", "convert", "specs"):
        path = tmp_path / (name + ".py")
        path.write_text("fixture-only " + name)
        files[name] = {"path": str(path), "sha256": sha(path)}
    original = dict(
        schema=PUBLIC,
        status="registered",
        mode="proof",
        first=10,
        deadline=610,
        seed=20262905,
        immutable_extra={"retained": True},
        train=files["train"],
        native=files["native"],
        metadata_spec_helper=files["specs"],
        source_sha256={v["path"]: v["sha256"] for v in files.values()},
    )

    def put(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": sha(path)}

    parent_candidate = {"path": "parent.pt", "sha256": "parent"}
    parent_c = {
        "generation": 0,
        "search_helper": {"path": "search.py", "sha256": "search"},
    }
    parent_s = {"generation": 1, "parent_candidate": parent_candidate}
    ps = put("parent-seal.json", parent_s)
    pc = put("parent-contract.json", parent_c)
    bridge = tmp_path / "parent_bridge.py"
    bridge.write_text(
        "import json\nfrom pathlib import Path\n"
        "def validate_admission_result(s,r):\n"
        "    return json.loads(Path(s['path']).read_text()), "
        "json.loads(Path(r['path']).read_text())\n"
    )
    data = put("dataset.json", {"rows": []})
    original.update(
        operator_end_epoch=1000,
        dataset=data,
        parent_candidate=dict(
            parent_candidate,
            contract_sha256=hashlib.sha256(
                json.dumps(parent_c, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        ),
    )
    helpers = {files[k]["path"]: files[k]["sha256"] for k in ("prove", "specs")}
    helpers[str(bridge)] = sha(bridge)
    c = dict(
        phase="own-learning",
        updates=64,
        root_source_type="procedural-uniform-legal-walk-v2",
        execution_scope_schema="human-prior-own-execution-contract-v1",
        execution_mode="proof",
        seed=20262905,
        original_first_epoch=10,
        original_deadline_epoch=610,
        operator_end_epoch=1000,
        dataset_sha256=data["sha256"],
        execution_helpers_sha256=helpers,
        parent_admission_seal=ps,
        parent_admission_result=pc,
        parent_candidate=parent_candidate,
        generation=1,
        parent_generation=0,
        bootstrap_candidate_path="parent.pt",
        bootstrap_candidate_sha256="parent",
        search_helper=parent_c["search_helper"],
        teacher_labels_used_in_own_phase=False,
    )
    original["contract"] = put("contract.json", c)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(original))
    return {"path": str(path), "sha256": sha(path)}, files["prove"]


def test_projection_preserves_every_original_field_except_declared_schema(tmp_path):
    reg, proof = fixture(tmp_path)
    original, projected, _, _ = project(reg, proof)
    assert projected["schema"] == EXECUTOR
    assert projected["registration_sha256"] == reg["sha256"]
    assert {k: projected[k] for k in original if k != "schema"} == {
        k: v for k, v in original.items() if k != "schema"
    }
    assert json.loads((tmp_path / "registration.json").read_text())["schema"] == PUBLIC


@pytest.mark.parametrize("mutation", ["old-schema", "input-SHA", "proof-SHA", "origin"])
def test_projection_rejects_wrong_public_identity(tmp_path, mutation):
    reg, proof = fixture(tmp_path)
    if mutation == "old-schema":
        path = tmp_path / "registration.json"
        raw = json.loads(path.read_text())
        raw["schema"] = EXECUTOR
        path.write_text(json.dumps(raw))
        reg["sha256"] = sha(path)
    elif mutation == "input-SHA":
        reg["sha256"] = "0" * 64
    elif mutation == "proof-SHA":
        proof["sha256"] = "0" * 64
    else:
        other = tmp_path / "other"
        other.mkdir()
        path = other / "prove.py"
        path.write_text("fixture-only prove")
        proof = {"path": str(path), "sha256": sha(path)}
    with pytest.raises(ValueError):
        project(reg, proof)


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "teacher-bootstrap"),
        ("original_deadline_epoch", 611),
        ("generation", 2),
    ],
)
def test_phase_clock_and_parent_generation_cannot_change(tmp_path, field, value):
    reg, proof = fixture(tmp_path)
    path = tmp_path / "contract.json"
    c = json.loads(path.read_text())
    c[field] = value
    path.write_text(json.dumps(c))
    reg_path = tmp_path / "registration.json"
    original = json.loads(reg_path.read_text())
    original["contract"]["sha256"] = sha(path)
    reg_path.write_text(json.dumps(original))
    reg["sha256"] = sha(reg_path)
    with pytest.raises(ValueError):
        project(reg, proof)
