"""Pure tiny fixtures only; no actor, NN, games or training subprocess launched."""

import json
import os
import sys
from pathlib import Path

import pytest
from controller_support import END, command, read_bound, sha, validate_clock


def fixture(tmp_path):
    config = {"max_actions": 8192, "original_deadline_epoch": END - 1}
    manifest = {
        "config": {"path": "CONFIG", "sha256": "CFG_SHA"},
        "checkout": "CLEAN_SOURCE",
        "model": {"path": "MODEL"},
        "anchor_model": {"path": "E8"},
        "helpers": {
            name: {"path": name}
            for name in ("produce_v2.py", "search.py", "value.py", "anchor_value.py")
        },
    }
    parent = tmp_path / "parent.gz"
    parent.write_bytes(b"synthetic-parent-no-actor")
    return manifest, config, parent


def test_actual_producer_interface_whole_and_resume_has_no_synthetic_mode(tmp_path):
    manifest, config, parent = fixture(tmp_path)
    whole = command(manifest, config, 1024, tmp_path / "whole.gz")
    resumed = command(manifest, config, 2048, tmp_path / "next.gz", parent, sha(parent))
    assert whole[:2] == [sys.executable, "produce_v2.py"]
    flags = dict(zip(resumed[2::2], resumed[3::2], strict=False))
    assert flags["--config-sha256"] == "CFG_SHA"
    assert flags["--resume-sha256"] == sha(parent) and flags["--resume"] == str(parent)
    assert (
        flags["--target-actions"] == "2048"
        and float(flags["--original-deadline"]) == config["original_deadline_epoch"]
    )
    assert "--synthetic-zero-fixture" not in whole + resumed


def test_changed_parent_sha_and_missing_parent_bind_rejected(tmp_path):
    manifest, config, parent = fixture(tmp_path)
    digest = sha(parent)
    parent.write_bytes(b"changed")
    with pytest.raises(ValueError, match="parent checkpoint SHA"):
        command(manifest, config, 2048, tmp_path / "next", parent, digest)
    with pytest.raises(ValueError, match="together"):
        command(manifest, config, 2048, tmp_path / "next", parent, None)
    with pytest.raises(ValueError, match="fixed epoch"):
        command(manifest, config, 8193, tmp_path / "next")


def test_expired_future_reset_and_operator_ceiling_rejected():
    assert validate_clock(100, 700, 600, 101) is None
    for first, deadline, now in [
        (100, 700, 700),
        (200, 800, 100),
        (100, 701, 101),
        (END - 599, END + 1, END - 598),
    ]:
        with pytest.raises(ValueError):
            validate_clock(first, deadline, 600, now)


def test_changed_config_sha_rejected(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"seed": 1}))
    digest = sha(config)
    assert read_bound(config, digest) == {"seed": 1}
    config.write_text(json.dumps({"seed": 2}))
    with pytest.raises(ValueError, match="input SHA"):
        read_bound(config, digest)


def test_commands_parser_options_bind_actual_producer_source():
    import ast

    path = Path("/workspace/HarbiChess/experiments/ufuk/cpu-fresh-selfplay-v2/produce_v2.py")
    tree = ast.parse(path.read_text())
    options = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    for name in [
        "--config",
        "--config-sha256",
        "--model",
        "--anchor-model",
        "--anchor-helper",
        "--source-repo",
        "--search-helper",
        "--value-helper",
        "--output",
        "--target-actions",
        "--original-deadline",
        "--resume",
        "--resume-sha256",
    ]:
        assert name in options


def test_owned_child_cleanup_if_command_receipt_publish_fails(tmp_path, monkeypatch):
    import controller_support as support

    class FakeProcess:
        pid = os.getpid()
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -15

    process = FakeProcess()
    monkeypatch.setattr(support.subprocess, "Popen", lambda *args, **kwargs: process)
    signals = []
    monkeypatch.setattr(support.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    published = []

    def publish(path, receipt):
        if path.name.endswith("command.json"):
            raise RuntimeError("synthetic atomic publication failure")
        published.append((path, receipt.copy()))

    with pytest.raises(RuntimeError, match="synthetic atomic"):
        support.guarded_child(
            ["NO_REAL_PROCESS"],
            {"checkout": str(tmp_path)},
            END,
            lambda deadline: None,
            tmp_path,
            "fixture",
            publish,
        )
    assert len(signals) == 1 and signals[0][0] == process.pid
    assert process.returncode == -15 and published[0][1]["status"] == "failed-preserved"
