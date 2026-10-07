"""Metadata-only original-producer/new-converter integration regression."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FACTORY = load(ROOT / "integration" / "seal_factory.py", "factory_v3_fixture")


def fixture(tmp_path):
    producer = tmp_path / "original-producer"
    producer.mkdir()
    (producer / "collector.py").write_text("original producer fixture")
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "model.py").write_text("feature fixture")
    prior = parent / "prior.py"
    prior.write_text("prior fixture")
    output = tmp_path / "output"
    output.mkdir()
    events = output / "events.jsonl"
    events.write_text("fixture only; no games")
    sources = {"collector.py": FACTORY.sha(producer / "collector.py")}
    reg = {"schema": "own-nnue-closed-terminal-collection-registration-v1",
           "status": "registered", "seed": 20262906, "output_path": str(output),
           "parent_helpers": {"directory": str(parent), "prior_path": str(prior)},
           "producer_source_sha256": sources, "operator_end_epoch": 1791448916.685839}
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps(reg))
    receipt = {"schema": "own-nnue-closed-terminal-collection-receipt-v1",
               "status": "PASS-exact-closed-terminal-row-budget", "seed": 20262906,
               "registration_sha256": FACTORY.sha(registration),
               "events_bytes": events.stat().st_size, "events_sha256": FACTORY.sha(events),
               "producer_source_sha256": sources, "alias_chunks": []}
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    return registration, receipt_path, producer


def build(tmp_path, mutation=None):
    reg, receipt, producer = fixture(tmp_path)
    converter = ROOT / "source" / "convert.py"
    helper = FACTORY.ref(converter)
    if mutation == "helper-sha":
        helper["sha256"] = "0" * 64
    elif mutation == "helper-path":
        helper = FACTORY.ref(ROOT / "source" / "contract.py")
    elif mutation == "producer":
        (producer / "collector.py").write_text("changed producer")
    return FACTORY.make_conversion_spec(
        reg, receipt, first=100, deadline=700, core_repo=tmp_path,
        producer_directory=producer, converter_helper=helper,
    )


def test_original_producer_and_corrective_converter_are_separate(tmp_path):
    spec = build(tmp_path)
    assert spec["producer_directory"] == str(tmp_path / "original-producer")
    assert spec["converter_helper"] == FACTORY.ref(ROOT / "source" / "convert.py")
    assert spec["first"] == 100 and spec["deadline"] == 700


@pytest.mark.parametrize("mutation", ["helper-sha", "helper-path", "producer"])
def test_wrong_sources_rejected(tmp_path, mutation):
    with pytest.raises(ValueError):
        build(tmp_path, mutation)


def test_converter_rejects_missing_explicit_binding_before_data_access():
    converter = load(ROOT / "source" / "convert.py", "converter_v3_fixture")
    with pytest.raises(ValueError, match="corrective converter"):
        converter.convert({"schema": "NNUE-closedterminal1024-conversion-seal-v1"})


def test_auditor_loads_explicit_converter_not_producer_converter():
    tree = ast.parse((ROOT / "integration" / "audit_six.py").read_text())
    loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_load_actual")
    assignment = next(n for n in loader.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "converter"
                              for t in n.targets))
    assert ast.unparse(assignment.value.args[0]) == "spec['converter_helper']"
    assert "directory / 'convert.py'" not in ast.unparse(loader)


def test_loaded_converter_rejects_different_self_path(tmp_path):
    converter = load(ROOT / "source" / "convert.py", "converter_v3_self_fixture")
    other = tmp_path / "convert.py"
    other.write_bytes((ROOT / "source" / "convert.py").read_bytes())
    with pytest.raises(ValueError, match="corrective converter"):
        converter.convert({"schema": "NNUE-closedterminal1024-conversion-seal-v1",
                           "converter_binding_schema":
                           "closed-terminal-corrective-converter-binding-v3",
                           "converter_helper": {"path": str(other), "sha256":
                                                hashlib.sha256(other.read_bytes()).hexdigest()}})
