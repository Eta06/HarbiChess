"""Pure wiring/identity tests; no real bank/model/search/training subprocess."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gencontrol_test", HERE / "control.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def test_publish_once_never_overwrites(tmp_path):
    p = tmp_path / "one.json"
    c.publish(p, dict(a=1))
    with pytest.raises(FileExistsError):
        c.publish(p, dict(a=2))
    assert json.loads(p.read_bytes()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp-*"))


def test_pid_reuse_never_signalled(monkeypatch):
    record = dict(pid=123, startticks=10, state="S")
    monkeypatch.setattr(c, "proc_identity", lambda pid: dict(pid=pid, startticks=11, state="S"))
    monkeypatch.setattr(c.os, "pidfd_open", lambda pid: pytest.fail("must not signal reused PID"))
    c.stop_owned({123: record})


def test_existing_owner_uses_pidfd_and_no_group_signal(monkeypatch):
    record = dict(pid=123, startticks=10, state="S")
    live = {123: True}
    sent = []
    monkeypatch.setattr(c, "same_owner", lambda r: live[r["pid"]])
    monkeypatch.setattr(c.os, "pidfd_open", lambda pid: 42)
    monkeypatch.setattr(c.os, "close", lambda fd: None)

    def signal(fd, sig):
        sent.append((fd, sig))
        live[123] = False

    monkeypatch.setattr(c.signal, "pidfd_send_signal", signal)
    c.stop_owned({123: record})
    assert sent == [(42, c.signal.SIGTERM)]


def test_registered_phase_preserves_one_clock_and_failure(tmp_path, monkeypatch):
    owner = c.Owner.__new__(c.Owner)
    owner.root = tmp_path
    owner.commands = []
    owner.guard = lambda end: None
    # Failure BEFORE any subprocess: recipe publishes original clock, no retry.
    seen = []

    def build(p, first, end):
        seen.append((first, end))
        raise ValueError("synthetic bad recipe")

    with pytest.raises(ValueError):
        owner.phase("phase", 600, build)
    result = json.loads((tmp_path / "phase/result.json").read_bytes())
    assert len(seen) == 1 and result["first"] == seen[0][0] and result["deadline"] == seen[0][1]
    assert result["deadline"] <= result["first"] + 600
    assert result["status"] == "FAILED-preserved"
    with pytest.raises(FileExistsError):
        owner.phase("phase", 600, build)


def test_bank_parser_flag_and_train_entry_wiring():
    text = (HERE / "control.py").read_text()
    assert "--registration" in text
    tree = ast.parse((HERE / "train_entry.py").read_text())
    prepare = [
        x
        for x in ast.walk(tree)
        if isinstance(x, ast.Call)
        and isinstance(x.func, ast.Attribute)
        and x.func.attr == "prepare"
    ]
    assert len(prepare) == 1 and len(prepare[0].args) == 3
    calls = [
        x
        for x in ast.walk(tree)
        if isinstance(x, ast.Call)
        and isinstance(x.func, ast.Attribute)
        and x.func.attr == "execute"
    ]
    assert len(calls) == 1 and len(calls[0].args) == 1
    assert "registration_sha256" in (HERE / "train_entry.py").read_text()


def test_six_bank_seed_schedule_not_old_selection():
    text = (HERE / "control.py").read_text()
    assert "seed + 1000003 * generation" in text
    assert "teacher-selected-ownq" not in text
    assert "for generation in (1, 2, 3)" in text
    assert "literalzero-generation0" in text


def test_actual_zero_metadata_preflight_uses_no_tensor_loader():
    text = (HERE / "control.py").read_text()
    assert ".validate_metadata(seal)" in text
    assert ".load_native(" not in text
    assert "torch.load" not in text
    assert "helper_inventory" in text and "source closure" in text


def test_sha_mutation_rejected(tmp_path):
    p = tmp_path / "x.json"
    c.publish(p, dict(a=1))
    r = c.ref(p)
    p.write_bytes(b"{}")
    with pytest.raises(ValueError):
        c.read(r)
