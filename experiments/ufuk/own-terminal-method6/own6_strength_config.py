"""Explicit SEARCH-ACTING-v2 qualification binding; imports alone never execute evaluations."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SEEDS = (20261625, 20261626)
BOOKS = {}
SOURCE = "428a30f5e1658f3cf159844db547ff0147ade5a9"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    assert config["schema"] == "ufuk-own6-strength-bindings-v1"
    assert config["status"] == "frozen-before-formal-execution"
    assert config["qualification_ledger_slot"] == 6 and config["MAX_families"] == 8
    assert config["seeds"] == list(SEEDS)
    books = config["books_sha256"]
    assert set(books) == {str(seed) for seed in SEEDS}
    assert len(set(books.values())) == 2
    assert all(
        len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in books.values()
    )
    assert config["books_provenance_sha256"] != "ROOT_UNKNOWN"
    assert len(config["books_provenance_sha256"]) == 64
    assert type(config["fixed_epochs"]) is int and config["fixed_epochs"] >= 8
    source = config["source_commit"]
    assert len(source) == 40 and all(c in "0123456789abcdef" for c in source)
    assert source == SOURCE
    assert config["native_schema"] == "torch-search-acting-native-cuda-v2"
    assert config["games_per_arm"] == 96 and config["arm_wall_seconds"] == 3600
    assert config["runtime_scope"] == "same-A100-CPU-one-thread-Torch2.11-all-six-arms"
    BOOKS.clear()
    BOOKS.update({int(k): v for k, v in books.items()})
    return config


def bind_cli(namespace):
    if not __debug__:
        raise RuntimeError("Required assertions disabled")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qualification-config", type=Path, required=True)
    parser.add_argument("--qualification-config-sha256", required=True)
    args, remaining = parser.parse_known_args()
    assert sha(args.qualification_config) == args.qualification_config_sha256
    config = validate_config(json.loads(args.qualification_config.read_text()))
    assert config["helper_sha256"][Path(namespace["__file__"]).name] == sha(
        namespace["__file__"]
    )
    package = importlib.util.find_spec("harbichess")
    assert package is not None and package.origin is not None
    checkout = Path(package.origin).resolve().parents[2]
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        == config["source_commit"]
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=checkout, text=True
    ).strip()
    namespace.update(Q=config, SEEDS=SEEDS, BOOKS=BOOKS)
    sys.argv = [sys.argv[0], *remaining]


def validate_binding(args, config):
    """All input manifests carry the actual new producer and fixed E explicitly."""
    if hasattr(args, "source_commit"):
        assert args.source_commit == config["source_commit"]
    for field in ("manifest", "registration"):
        if hasattr(args, field):
            value = json.loads(Path(getattr(args, field)).read_text())
            assert value["source_commit"] == config["source_commit"]
            assert value["fixed_epochs"] == config["fixed_epochs"]
            assert value["qualification_ledger_slot"] == 6
