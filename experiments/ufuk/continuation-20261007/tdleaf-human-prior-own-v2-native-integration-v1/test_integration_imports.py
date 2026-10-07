import hashlib
import importlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def test_repaired_proof_import_and_local_contract_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    for name in ("prove", "convert", "native", "contracts", "train"):
        sys.modules.pop(name, None)
    prove = importlib.import_module("prove")
    convert = importlib.import_module("convert")
    assert prove.module is convert.load_module
    assert callable(prove.execute)
    assert callable(convert.pins_tree)
    assert callable(importlib.import_module("contracts").build)
    assert callable(importlib.import_module("native").load_native)


def test_pins_tree_checks_every_nested_input(tmp_path):
    monkeypatch_path = tmp_path / "input.json"
    monkeypatch_path.write_bytes(b"sealed")
    ref = {"path": str(monkeypatch_path), "sha256": hashlib.sha256(b"sealed").hexdigest()}
    sys.path.insert(0, str(HERE))
    try:
        convert = importlib.import_module("convert")
        convert.pins_tree({"nested": [ref]})
        monkeypatch_path.write_bytes(b"changed")
        with pytest.raises(ValueError):
            convert.pins_tree({"nested": [ref]})
    finally:
        sys.path.remove(str(HERE))
        sys.modules.pop("convert", None)
