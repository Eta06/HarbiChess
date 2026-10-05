"""Synthetic archive fixtures only: no actual sealed capsule is packed here."""

import gzip
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import cpu_deduplicated_backup as backup
import pytest


def fixture_archive(tmp_path, *, link=False, corrupt=False):
    native = Path("/original/run/checkpoints/epoch-00000001")
    actor = {
        "epoch": 1,
        "replay_buffer": "closed-empty",
        "training_pass": "closed",
        "pending_search_schedule": "closed-empty",
        "sample_chain_sha256": "fixture-chain",
    }
    journal = gzip.compress(
        json.dumps(
            {"schema": "fixture-run", "epoch": 1, "sample_chain_sha256": "fixture-chain"}
        ).encode()
    )
    values = {
        str(native / name): b"fixture exact model Adam RNG bytes\x00\x80" for name in backup.FILES
    }
    values[str(native / "actor.json")] = json.dumps(actor).encode()
    values[str(native / "last-frozen-epoch.json.gz")] = journal
    values[str(native.parent.parent / "journal/epoch-00000001.json.gz")] = journal
    inputs = {}
    for key in ("book", "initial_weights", "experiment_config", "protocol"):
        value = ("fixture-" + key).encode()
        path = Path("/original/inputs") / key
        values[str(path)] = value
        inputs[key] = {
            "relative_path": "../../../inputs/" + key,
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    m = {
        "schema": "torch-search-acting-native-cpu-v3",
        "source_commit": backup.SOURCE,
        "runtime": {"device": "cpu"},
        "state": actor,
        "inputs": inputs,
        "run_config": {
            "config": {"device": "cpu"},
            "schema": "fixture-run",
            "input_sha256": {key: item["sha256"] for key, item in inputs.items()},
        },
        "artifacts": {
            name: hashlib.sha256(values[str(native / name)]).hexdigest() for name in backup.FILES
        },
    }
    values[str(native / "checkpoint.json")] = json.dumps(m).encode()
    rows = {
        path: {
            "path": path,
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
            "reviewed_public": True,
        }
        for path, value in values.items()
    }
    entries, _ = backup.make_blob_entries(list(rows), rows)
    source = b"not-a-real-source-bundle: test-only fixture"
    ledger = {
        "schema": "cpu-v3-content-addressed-regular-blob-map-v1",
        "source_commit": backup.SOURCE,
        "files": entries,
        "native_manifests": {str(native): m},
        "source_bundle_bytes": len(source),
        "source_bundle_sha256": hashlib.sha256(source).hexdigest(),
    }
    blobs = {row["member"]: values[row["path"]] for row in entries}
    blobs["blobs/" + ledger["source_bundle_sha256"]] = source
    archive = tmp_path / "synthetic.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        data = json.dumps(ledger).encode()
        info = tarfile.TarInfo("artifact-map.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        for index, (name, value) in enumerate(blobs.items()):
            info = tarfile.TarInfo(name)
            if link and index == 0:
                info.type = tarfile.LNKTYPE
                info.linkname = "other"
                tar.addfile(info)
            else:
                if corrupt and index == 0:
                    value = b"x" * len(value)
                info.size = len(value)
                tar.addfile(info, io.BytesIO(value))
    return archive, native, values, rows


def test_dedup_stores_one_regular_blob_and_restores_all_original_native_bytes(tmp_path):
    archive, native, values, rows = fixture_archive(tmp_path)
    entries, blobs = backup.make_blob_entries(list(rows), rows)
    assert len(entries) == len(values)
    assert len(blobs) < len(entries)
    result = backup.restore_selected(
        archive, backup.sha(archive), tmp_path / "fresh", native, time.time() + 20
    )
    assert result["status"] == "bytes-pass"
    assert result["native_runtime_loaded"] is False
    for original, expected in values.items():
        restored = tmp_path / "fresh" / Path(original).relative_to("/")
        assert restored.read_bytes() == expected
    with pytest.raises(ValueError, match="fresh-restore-prefix"):
        backup.restore_selected(
            archive, backup.sha(archive), tmp_path / "fresh", native, time.time() + 20
        )


@pytest.mark.parametrize("kind", ["link", "corrupt"])
def test_restore_rejects_link_and_hash_changed_regular_blob(tmp_path, kind):
    archive, native, _, _ = fixture_archive(tmp_path, **{kind: True})
    with pytest.raises(ValueError, match="unexpected-link|blob-sha-mismatch"):
        backup.restore_selected(
            archive, backup.sha(archive), tmp_path / "fresh", native, time.time() + 20
        )


def test_same_sha_conflicting_size_is_rejected():
    digest = "a" * 64
    rows = {"/a": {"sha256": digest, "bytes": 1}, "/b": {"sha256": digest, "bytes": 2}}
    with pytest.raises(ValueError, match="conflicting-size"):
        backup.make_blob_entries(list(rows), rows)
