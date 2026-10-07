import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
import seal_factory  # noqa: E402


def dump(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_make_conversion_seal_pins_completed_original_ownq_v2(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    parent = source_dir / "teacher.pt"
    parent.write_bytes(b"synthetic parent bytes")
    features = source_dir / "model.py"
    features.write_text("synthetic feature helper")
    prior = source_dir / "prior.py"
    prior.write_text("synthetic prior helper")
    source_converter = source_dir / "convert.py"
    source_converter.write_text("synthetic old converter")
    dataset = source_dir / "dataset.json"
    dataset.write_text("{}\n")
    provenance = source_dir / "provenance.json"
    provenance.write_text("{}\n")
    registration_path = source_dir / "registration.json"
    registration = {
        "schema": "own-nnue-ownq-collection-registration-v2",
        "status": "registered",
        "seed": 20262905,
        "parent_candidate": {
            "path": str(parent.resolve()),
            "sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        },
    }
    registration_ref = dump(registration_path, registration)
    receipt_path = source_dir / "receipt.json"
    receipt = {
        "schema": "own-nnue-ownq-collection-receipt-v2",
        "status": "PASS-exact-row-budget",
        "train_rows": 1024,
        "training_row_ids": [f"g{i}:0" for i in range(1024)],
    }
    receipt_ref = dump(receipt_path, receipt)
    source_prov = {
        "collection_receipt_sha256": receipt_ref["sha256"],
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    dump(provenance, source_prov)
    result_path = source_dir / "result.json"
    dump(
        result_path,
        {
            "status": "PASS-own1024-fullhistory-trace-conversion-not-strength",
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
        },
    )
    source_spec = {
        "schema": "NNUE-own1024-dataset-conversion-seal-v2",
        "registration": registration_ref,
        "receipt": receipt_ref,
        "features": {
            "path": str(features.resolve()),
            "sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
        },
        "prior": {
            "path": str(prior.resolve()),
            "sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
        },
    }
    spec_path = source_dir / "conversion-seal.json"
    dump(spec_path, source_spec)
    spec = seal_factory.make_conversion_spec(
        spec_path,
        result_path,
        first=1000.0,
        deadline=1500.0,
        operator_end_epoch=1791448916.685839,
        source_converter_path=source_converter,
        core_repo=tmp_path,
    )
    assert spec["schema"] == "NNUE-own-afterstate1024-conversion-seal-v1"
    assert spec["seed"] == 20262905
    assert spec["source_conversion_seal"] == seal_factory.ref(spec_path)
    assert spec["parent_candidate"] == registration["parent_candidate"]


def test_phase_clock_requires_new_bounded_root_clock():
    with pytest.raises(ValueError, match="explicit ROOT phase clock"):
        seal_factory._phase_clock("proof", 1000.0, 1601.0, 2000.0)
    with pytest.raises(ValueError, match="operator window"):
        seal_factory._phase_clock("fresh-fit", 1000.0, 2000.0, 1791448916.685840)
    seal_factory._phase_clock("fresh-fit", 1000.0, 2000.0, 1791448916.685839)
