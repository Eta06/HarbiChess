import gzip
import hashlib
import json

import pytest

from harbichess.training.torch_online_run import _acquire_lock, _journal, _verify_journals


def record(step, previous):
    row = {"schema": "unit-journal", "update": step, "samples": [{"action": "e2e4"}]}
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    chain = hashlib.sha256(bytes.fromhex(previous) + encoded).hexdigest()
    return row | {"previous_sample_chain_sha256": previous, "sample_chain_sha256": chain}


def test_whole_journal_prefix_restores_and_published_bytes_are_not_overwritten(tmp_path):
    (tmp_path / "journal").mkdir()
    chain = hashlib.sha256(b"").hexdigest()
    for step in range(1, 3):
        row = record(step, chain)
        path = tmp_path / "journal" / f"update-{step:08d}.json.gz"
        _journal(path, row)
        before, modified = path.read_bytes(), path.stat().st_mtime_ns
        _journal(path, row)
        assert path.read_bytes() == before and path.stat().st_mtime_ns == modified
        with pytest.raises(ValueError, match="differs"):
            _journal(path, row | {"samples": []})
        assert path.read_bytes() == before
        chain = row["sample_chain_sha256"]
    _verify_journals(tmp_path, 2, chain)
    path = tmp_path / "journal/update-00000001.json.gz"
    corrupted = json.loads(gzip.decompress(path.read_bytes()))
    corrupted["samples"][0]["action"] = "d2d4"
    path.write_bytes(gzip.compress(json.dumps(corrupted).encode()))
    with pytest.raises(ValueError, match="content hash"):
        _verify_journals(tmp_path, 2, chain)


def test_cursor_hash_and_missing_history_cannot_be_silently_accepted(tmp_path):
    (tmp_path / "journal").mkdir()
    row = record(1, hashlib.sha256(b"").hexdigest())
    _journal(tmp_path / "journal/update-00000001.json.gz", row)
    with pytest.raises(ValueError, match="native checkpoint"):
        _verify_journals(tmp_path, 1, "0" * 64)
    with pytest.raises(FileNotFoundError):
        _verify_journals(tmp_path, 2, row["sample_chain_sha256"])


def test_interrupted_temporary_bytes_remain_preserved_during_new_atomic_publish(tmp_path):
    old = tmp_path / ".update-00000001.json.gz.old.tmp"
    old.write_bytes(b"retained interrupted bytes")
    row = record(1, hashlib.sha256(b"").hexdigest())
    _journal(tmp_path / "update-00000001.json.gz", row)
    assert old.read_bytes() == b"retained interrupted bytes"


def test_live_run_owner_blocks_second_invocation_without_changing_owner(tmp_path):
    path, owner = _acquire_lock(tmp_path, "a" * 40)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="live online"):
        _acquire_lock(tmp_path, "a" * 40)
    assert path.read_bytes() == before and json.loads(before) == owner
    path.unlink()
