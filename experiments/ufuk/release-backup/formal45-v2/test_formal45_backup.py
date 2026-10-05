import gzip
import hashlib
import io
import json
import os
import tarfile

import formal45_release_backup as backup
import pytest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    content = tmp_path / "content"
    runs, inputs, repo = (content / name for name in ("runs", "inputs", "repo"))
    for path in (runs, inputs, repo):
        path.mkdir(parents=True)
    monkeypatch.setattr(backup, "RUNS", runs)
    monkeypatch.setattr(backup, "CONTENT_ROOT", content)
    monkeypatch.setattr(backup, "INPUT_ROOTS", (inputs,))
    output = runs / "packages"
    monkeypatch.setattr(backup, "OUTPUT_ROOT", output)
    slot, seed = 4, 20261425
    method = backup.KNOWN_METHODS[slot]
    run = runs / method["run_template"].format(seed=seed)
    (run / "journal").mkdir(parents=True)
    (run / "metadata.json").write_text('{"original":"never changed"}')
    source = method["source"]
    protocol = inputs / "registration.json"
    protocol.write_text(
        json.dumps(
            {
                "schema": method["registration_schema"],
                "source_commit": source,
                "qualification_ledger_slot": slot,
                "status": "frozen-before-formal-execution",
            }
        )
    )
    long = inputs / ("a" * 75) / ("b" * 65)
    long.mkdir(parents=True)
    paths = {
        "protocol": protocol,
        "book": inputs / "book.json",
        "experiment_config": long / "config.json",
        "initial_weights": inputs / "e8.bin",
    }
    for key, path in paths.items():
        if key != "protocol":
            path.write_bytes(key.encode())
    evidence = inputs / "failed-preserved.json"
    evidence.write_text('{"status":"failed-preserved"}')
    job = {
        "slot": slot,
        "source_commit": source,
        "source_repo": str(repo),
        "protocol_sha256": digest(protocol),
        "reviewed_public_evidence": [
            {
                "path": str(evidence),
                "sha256": digest(evidence),
                "purpose": "failed-partial",
                "reviewed_public": True,
            }
        ],
    }
    monkeypatch.setattr(
        backup, "JOB_SPEC", {"verified_spec_sha256": "d" * 64, "jobs": {str(seed): job}}
    )
    monkeypatch.setattr(backup, "_assert_source_repo", lambda *_: None)

    def bundle(_repo, _source, destination):
        destination.write_bytes(b"test-only bundle bytes, no Git operation")
        return digest(destination)

    monkeypatch.setattr(backup, "_source_bundle", bundle)
    manifests = []
    for epoch in (0, 1):
        native = run / f"checkpoints/epoch-{epoch:08d}"
        native.mkdir(parents=True)
        state = {
            "epoch": epoch,
            "replay_buffer": "closed-empty",
            "training_pass": "closed",
            "pending_search_schedule": "closed-empty",
            "sample_chain_sha256": str(epoch) * 64,
        }
        actor = json.dumps(state).encode()
        record = {
            "epoch": 1,
            "schema": "torch-fresh-sparse-ownsearch-v1",
            "previous_sample_chain_sha256": "0" * 64,
            "sample_chain_sha256": "1" * 64,
        }
        journal = gzip.compress(json.dumps(record).encode(), mtime=0) if epoch else b""
        for name in backup.NATIVE_FILES:
            (native / name).write_bytes(
                actor
                if name == "actor.json"
                else journal
                if name == "last-frozen-epoch.json.gz"
                else name.encode()
            )
        if epoch:
            (run / "journal/epoch-00000001.json.gz").write_bytes(journal)
        manifest = {
            "schema": method["native_schema"],
            "source_commit": source,
            "state": state,
            "artifacts": {name: digest(native / name) for name in backup.NATIVE_FILES},
            "run_config": {"schema": "torch-fresh-sparse-ownsearch-v1", "config": {"seed": seed}},
            "inputs": {
                key: {"relative_path": os.path.relpath(path, native), "sha256": digest(path)}
                for key, path in paths.items()
            },
        }
        (native / "checkpoint.json").write_text(json.dumps(manifest))
        manifests.append(native / "checkpoint.json")
    return seed, run, evidence, manifests


def test_closed_prefix_all_bytes_long_inputs_and_verified_journal_alias(setup):
    seed, run, evidence, _ = setup
    originals = {str(p): digest(p) for p in run.rglob("*") if p.is_file()}
    package = backup.pack_seed(seed, base_url="https://example.trycloudflare.com")
    parsed = json.loads((package / "package.json").read_text())
    assert parsed["max_closed_epoch"] == 1 and len(parsed["archives"]) == 3
    epoch1 = next(row for row in parsed["archives"] if row["epoch"] == 1)
    members = backup.verify_archive(package / epoch1["name"])
    alias = members["checkpoints/epoch-00000001/last-frozen-epoch.json.gz"]
    assert alias["kind"] == "verified-journal-alias"
    assert alias["sha256"] == digest(run / "journal/epoch-00000001.json.gz")
    metadata = next(row for row in parsed["archives"] if row["epoch"] is None)
    members = backup.verify_archive(package / metadata["name"])
    assert "metadata/inputs/experiment_config.bin" in members
    assert members["metadata/evidence/000000.bin"]["sha256"] == digest(evidence)
    assert originals == {str(p): digest(p) for p in run.rglob("*") if p.is_file()}
    assert len(backup.build_server_map(package)) == 3


@pytest.mark.parametrize("change", ["schema", "source", "boundary"])
def test_wrong_method_native_or_open_boundary_fails_closed(setup, change):
    seed, _, _, manifests = setup
    path = manifests[0]
    data = json.loads(path.read_text())
    if change == "schema":
        data["schema"] = "torch-fullgame-native-cuda-v1"
    elif change == "source":
        data["source_commit"] = backup.KNOWN_METHODS[5]["source"]
    else:
        data["state"]["pending_search_schedule"] = "not-empty"
    path.write_text(json.dumps(data))
    with pytest.raises(backup.BackupError):
        backup.pack_seed(seed, base_url="https://example.trycloudflare.com")


def test_unreviewed_or_changed_evidence_rejected(setup):
    seed, _, evidence, _ = setup
    evidence.write_text("changed since explicit review")
    with pytest.raises(backup.BackupError, match="evidence-file-sha256"):
        backup.pack_seed(seed, base_url="https://example.trycloudflare.com")


@pytest.mark.parametrize("kind", ["symlink", "traversal", "wrong-hardlink", "forward-hardlink"])
def test_archive_reader_never_blindly_extracts_links(tmp_path, kind):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("safe")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
        link = tarfile.TarInfo(
            "../escape"
            if kind == "traversal"
            else "checkpoints/epoch-00000001/last-frozen-epoch.json.gz"
        )
        link.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
        link.linkname = "journal/epoch-00000001.json.gz" if kind == "forward-hardlink" else "safe"
        archive.addfile(link)
    with pytest.raises(backup.BackupError):
        backup.verify_archive(path)
    assert not (tmp_path / "escape").exists()


def test_matching_immutable_package_retry_is_idempotent(setup):
    seed, _, _, _ = setup
    first = backup.pack_seed(seed, base_url="https://example.trycloudflare.com")
    before = {p.name: digest(p) for p in first.iterdir() if p.is_file()}
    second = backup.pack_seed(seed, base_url="https://example.trycloudflare.com")
    assert first == second
    assert before == {p.name: digest(p) for p in first.iterdir() if p.is_file()}


def test_closed_native_mutation_during_pack_is_rejected(setup, monkeypatch):
    seed, _, _, manifests = setup
    original = backup._write_tar_gz
    changed = False

    def mutate_after_write(*args, **kwargs):
        nonlocal changed
        original(*args, **kwargs)
        if not changed:
            manifests[0].write_text(manifests[0].read_text() + " ")
            changed = True

    monkeypatch.setattr(backup, "_write_tar_gz", mutate_after_write)
    with pytest.raises(backup.BackupError, match="closed-native-bytes-changed"):
        backup.pack_seed(seed, base_url="https://example.trycloudflare.com")


def test_only_two_committed_manifest_selectors(monkeypatch):
    from scripts import upload_allowlisted_a100_artifacts as upload

    monkeypatch.delenv("HARBICHESS_RELEASE_MANIFEST", raising=False)
    assert upload._manifest_path() == upload.MANIFEST
    monkeypatch.setenv("HARBICHESS_RELEASE_MANIFEST", "formal45-v2")
    assert upload._manifest_path() == upload.FORMAL45_MANIFEST
    monkeypatch.setenv("HARBICHESS_RELEASE_MANIFEST", "../../private-key")
    with pytest.raises(upload.DeliveryError, match="manifest-selector"):
        upload._manifest_path()


@pytest.mark.parametrize("violation", ["count", "bytes"])
def test_aggregate_prior_assets_preserved_and_original_caps_enforced(setup, violation):
    seed, _, _, _ = setup
    package = backup.pack_seed(seed, base_url="https://new.trycloudflare.com")
    prior_path = backup.INPUT_ROOTS[0] / "prior.json"
    prior = {
        "schema": "harbichess-a100-release-transport-v1",
        "release_id": backup.RELEASE_ID,
        "release_tag": backup.RELEASE_TAG,
        "seeds": [20261205, 20261206],
        "assets": [],
    }
    count = 199 if violation == "count" else 9
    for index in range(count):
        sha = f"{index + 1:064x}"
        name = f"sha256-{sha}.tar.gz"
        prior["assets"].append(
            {
                "name": name,
                "bytes": 1 if violation == "count" else 1024**3,
                "sha256": sha,
                "url": f"https://old.trycloudflare.com/{name}",
            }
        )
    prior_path.write_text(json.dumps(prior))
    backup.JOB_SPEC["prior_transport_manifests"] = [
        {"path": str(prior_path), "sha256": digest(prior_path)}
    ]
    with pytest.raises(backup.BackupError, match="asset-count|total-exceeds"):
        backup.build_transport_manifest([package], "https://new.trycloudflare.com")


def test_prior_public_assets_keep_their_existing_urls(setup):
    seed, _, _, _ = setup
    package = backup.pack_seed(seed, base_url="https://new.trycloudflare.com")
    prior_path = backup.INPUT_ROOTS[0] / "prior.json"
    sha = "c" * 64
    asset = {
        "name": f"sha256-{sha}.tar.gz",
        "bytes": 3,
        "sha256": sha,
        "url": f"https://old.trycloudflare.com/sha256-{sha}.tar.gz",
    }
    prior_path.write_text(
        json.dumps(
            {
                "schema": "harbichess-a100-release-transport-v1",
                "release_id": backup.RELEASE_ID,
                "release_tag": backup.RELEASE_TAG,
                "seeds": [20261205, 20261206],
                "assets": [asset],
            }
        )
    )
    backup.JOB_SPEC["prior_transport_manifests"] = [
        {"path": str(prior_path), "sha256": digest(prior_path)}
    ]
    manifest = backup.build_transport_manifest([package], "https://new.trycloudflare.com")
    assert asset in manifest["assets"]
    assert len(manifest["assets"]) == 4
    assert manifest["seeds"] == [20261205, 20261206, seed]
