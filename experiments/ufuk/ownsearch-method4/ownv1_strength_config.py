"""Explicit OWN-v1 qualification binding; imports alone never execute evaluations."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SEEDS = (20261425, 20261426)
BOOKS = {
    20261425: "3fcb0d8e7542f93a093de4387a0cb15eb2862dcfc72291cab22832c4847e16ca",
    20261426: "39f073aa0f8a79431550b3295ab292518a8f0229c16910c474166340af5b3cff",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    assert config["schema"] == "ufuk-ownv1-strength-bindings-v1"
    assert config["status"] == "frozen-before-formal-execution"
    assert config["qualification_ledger_slot"] == 4 and config["MAX_families"] == 8
    assert config["seeds"] == list(SEEDS)
    assert config["books_sha256"] == {str(k): v for k, v in BOOKS.items()}
    assert type(config["fixed_epochs"]) is int and config["fixed_epochs"] >= 8
    source = config["source_commit"]
    assert len(source) == 40 and all(c in "0123456789abcdef" for c in source)
    assert source != "2312652dc52a894e9726f48321117cf114270355"
    assert config["native_schema"] == "torch-ownsearch-native-cuda-v1"
    assert config["games_per_arm"] == 96 and config["arm_wall_seconds"] == 3600
    assert config["runtime_scope"] == "same-A100-CPU-one-thread-Torch2.11-all-six-arms"
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
    assert config["helper_sha256"][Path(namespace["__file__"]).name] == sha(namespace["__file__"])
    package = importlib.util.find_spec("harbichess")
    assert package is not None and package.origin is not None
    checkout = Path(package.origin).resolve().parents[2]
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
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
            assert value["qualification_ledger_slot"] == 4
