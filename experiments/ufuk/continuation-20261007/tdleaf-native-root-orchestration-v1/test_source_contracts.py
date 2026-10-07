import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_refs_flattens_nested_refs_by_canonical_path():
    controller = load("tdleaf_controller_test", HERE / "controller.py")
    ref_a = {"path": "/tmp/a", "sha256": "a" * 64}
    ref_b = {"path": "/tmp/b", "sha256": "b" * 64}
    assert controller.collect_refs({"a": ref_a, "nested": [ref_b, ref_a]}) == {
        "/tmp/a": ref_a,
        "/tmp/b": ref_b,
    }


def test_collect_refs_rejects_conflicting_hashes_for_same_path():
    controller = load("tdleaf_controller_conflict_test", HERE / "controller.py")
    with pytest.raises(ValueError):
        controller.collect_refs(
            {"a": {"path": "/tmp/x", "sha256": "a" * 64},
             "b": {"path": "/tmp/x", "sha256": "b" * 64}}
        )


def test_collect_refs_canonicalizes_and_rejects_symlink_sources(tmp_path):
    controller = load("tdleaf_controller_paths_test", HERE / "controller.py")
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    digest = controller.sha(target)
    refs = controller.collect_refs({"one": {"path": str(target), "sha256": digest}})
    assert refs == {str(target.resolve()): {"path": str(target.resolve()), "sha256": digest}}
    with pytest.raises(ValueError, match="symlinks"):
        controller.collect_refs({"one": {"path": str(alias), "sha256": digest}})


def test_h0_adapter_rejects_symlink_refs(tmp_path):
    adapter = load("tdleaf_h0_symlink_adapter_test", HERE / "known160_h0_admission.py")
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="pinned regular"):
        adapter.pin({"path": str(alias), "sha256": adapter.sha(target)})


def test_known160_adapter_fails_closed_on_wrong_schema_or_teacher_role():
    adapter = load("tdleaf_h0_adapter_test", HERE / "known160_h0_admission.py")
    with pytest.raises(ValueError):
        adapter.admit_child(
            {"schema": "teacher-admission", "status": "registered", "seed": 20262905}
        )
    with pytest.raises(ValueError):
        adapter.admit_child(
            {"schema": "tdleaf-own-v2-h0-known160-child-v1", "status": "registered",
             "seed": 20262905, "teachers": {"20262905": {}}}
        )
