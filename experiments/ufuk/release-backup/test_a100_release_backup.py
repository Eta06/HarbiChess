import gzip
import hashlib
import http.client
import json
import os
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import a100_release_backup as backup


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_native(path, epoch, inputs, source, journal=b""):
    path.mkdir(parents=True)
    state = {
        "epoch": epoch,
        "replay_buffer": "closed-empty",
        "ppo_pass": "closed",
        "sample_chain_sha256": "a" * 64 if epoch else "0" * 64,
    }
    files = {
        "model.safetensors": b"model" + bytes([epoch]),
        "base.safetensors": b"base",
        "behavior.safetensors": b"behavior" + bytes([epoch]),
        "training.pt": b"training" + bytes([epoch]),
        "actor.json": (json.dumps(state, sort_keys=True) + "\n").encode(),
        "last-frozen-epoch.json.gz": journal,
    }
    for name, data in files.items():
        (path / name).write_bytes(data)
    manifest = {
        "schema": "torch-fullgame-native-cuda-v1",
        "source_commit": source,
        "state": state,
        "run_config": {
            "schema": "torch-fresh-fullgame-ppo-v1",
            "config": {"seed": 20261205, "epoch_steps": 1},
        },
        "base_model_sha256": "b" * 64,
        "inputs": inputs,
        "artifacts": {name: _hash(path / name) for name in backup.NATIVE_FILES},
    }
    (path / "checkpoint.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")


def _make_git_repo(root):
    root.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Scratch Test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "scratch@example.invalid"], cwd=root, check=True)
    reg = root / "registration.json"
    reg.write_text(json.dumps({"schema": "ufuk-method2-formal-registration-v3"}) + "\n")
    subprocess.run(["git", "add", "registration.json"], cwd=root, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "fixture"], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_pack_two_epochs_metadata_prefix_and_sha_names(tmp_path, monkeypatch):
    content = tmp_path / "content"
    input1 = content / "harbichess-inputs"
    input2 = content / "harbichess-fullgame-method2-inputs"
    input1.mkdir(parents=True)
    input2.mkdir(parents=True)
    inputs_files = {
        "initial_weights": input1 / "initial.safetensors",
        "book": input1 / "book.json",
        "experiment_config": input2 / "config.json",
        "protocol": input2 / "protocol.json",
    }
    for name, path in inputs_files.items():
        path.write_bytes(name.encode())
    repo = tmp_path / "repo"
    source = _make_git_repo(repo)
    runs = content / "harbichess-runs"
    run = runs / "fullgame-method2-seed-20261205" / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "journal").mkdir()
    native_inputs = {}
    for key, path in inputs_files.items():
        native_inputs[key] = {
            "relative_path": os.path.relpath(
                path.resolve(), (run / "checkpoints/epoch-00000000").resolve()
            ),
            "sha256": _hash(path),
        }
    _write_native(run / "checkpoints/epoch-00000000", 0, native_inputs, source)
    journal = gzip.compress(
        json.dumps({"epoch": 1, "sample_chain_sha256": "a" * 64}, sort_keys=True).encode(), mtime=0
    )
    (run / "journal/epoch-00000001.json.gz").write_bytes(journal)
    _write_native(run / "checkpoints/epoch-00000001", 1, native_inputs, source, journal)

    monkeypatch.setattr(backup, "CONTENT_ROOT", content)
    monkeypatch.setattr(backup, "RUNS", runs)
    monkeypatch.setattr(backup, "SOURCE_REPO", repo)
    monkeypatch.setattr(backup, "INPUT_ROOTS", (input1, input2))
    monkeypatch.setattr(backup, "REGISTRATION_REL", Path("registration.json"))
    monkeypatch.setattr(backup, "OUTPUT_ROOT", runs / "release-packages")
    package_dir = backup.pack_seed(20261205, base_url="https://test123.trycloudflare.com")
    package = json.loads((package_dir / "package.json").read_text())
    assert package["max_closed_epoch"] == 1
    assert len(package["archives"]) == 3
    assert all(a["name"] == f"sha256-{a['sha256']}.tar.gz" for a in package["archives"])
    transport = json.loads((package_dir / "transport-manifest.json").read_text())
    assert all(
        a["url"].startswith("https://test123.trycloudflare.com/") for a in transport["assets"]
    )
    combined = backup.build_transport_manifest([package_dir], "https://test123.trycloudflare.com")
    assert len(combined["assets"]) == 3 and combined["seeds"] == [20261205]

    second = backup.OUTPUT_ROOT / "second"
    second.mkdir()
    extra = b"second registered seed archive"
    extra_sha = hashlib.sha256(extra).hexdigest()
    extra_name = f"sha256-{extra_sha}.tar.gz"
    (second / extra_name).write_bytes(extra)
    (second / extra_name).chmod(0o600)
    (second / "package.json").write_text(
        json.dumps(
            {
                "schema": "harbichess-a100-release-package-v1",
                "seed": 20261206,
                "archives": [
                    {
                        "name": extra_name,
                        "path": extra_name,
                        "bytes": len(extra),
                        "sha256": extra_sha,
                    }
                ],
            }
        )
    )
    merged = backup.merge_server_maps([package_dir, second])
    assert len(merged) == 4
    combined = backup.build_transport_manifest(
        [package_dir, second], "https://test123.trycloudflare.com"
    )
    assert len(combined["assets"]) == 4 and combined["seeds"] == [20261205, 20261206]
    epoch1 = next(a for a in package["archives"] if a["epoch"] == 1)
    with tarfile.open(package_dir / epoch1["name"], "r:gz") as tar:
        journal_member = tar.getmember("journal/epoch-00000001.json.gz")
        frozen_member = tar.getmember("checkpoints/epoch-00000001/last-frozen-epoch.json.gz")
        assert frozen_member.islnk()
        assert frozen_member.linkname == journal_member.name
        assert tar.extractfile(frozen_member).read() == journal
    metadata = next(a for a in package["archives"] if a["epoch"] is None)
    with tarfile.open(package_dir / metadata["name"], "r:gz") as tar:
        names = set(tar.getnames())
        assert "metadata/source.bundle" in names
        assert "metadata/registration/registration.json" in names
        assert "restore-content/harbichess-inputs/book.json" in names
        assert "metadata/restore.txt" in names
        assert not any("logs" in name or ".ssh" in name for name in names)

    epoch0_manifest = run / "checkpoints/epoch-00000000/checkpoint.json"
    broken = json.loads(epoch0_manifest.read_text())
    broken["run_config"]["config"]["seed"] = 20261206
    epoch0_manifest.write_text(json.dumps(broken, sort_keys=True) + "\n")
    try:
        backup.pack_seed(20261205, base_url="https://test123.trycloudflare.com")
    except backup.BackupError as exc:
        assert "run-config-seed-mismatch" in str(exc)
    else:
        raise AssertionError("packager accepted checkpoint from another seed")


def test_readonly_server_serves_only_manifest_sha_files(tmp_path):
    data = b"tiny exact archive fixture"
    digest = hashlib.sha256(data).hexdigest()
    name = f"sha256-{digest}.tar.gz"
    path = tmp_path / name
    path.write_bytes(data)
    path.chmod(0o600)
    manifest = {
        "schema": "harbichess-a100-release-package-v1",
        "archives": [{"name": name, "path": name, "bytes": len(data), "sha256": digest}],
    }
    (tmp_path / "package.json").write_text(json.dumps(manifest))
    files = backup.build_server_map(tmp_path)
    server = backup.AllowlistHTTPServer(
        ("127.0.0.1", 0), backup.ReadOnlyHandler, files, time.time() + 60
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/" + name)
        response = conn.getresponse()
        assert response.status == 200
        assert int(response.getheader("Content-Length")) == len(data)
        assert response.read() == data
        conn.close()

        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("HEAD", "/" + name)
        response = conn.getresponse()
        assert response.status == 200 and response.read() == b""
        conn.close()

        for target in ("/", "/../package.json", "/" + name + "?x=1", "/%2e%2e/package.json"):
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", target)
            assert conn.getresponse().status == 404
            conn.close()
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("POST", "/" + name, body=b"overwrite")
        assert conn.getresponse().status == 405
        conn.close()

        expired = backup.AllowlistHTTPServer(
            ("127.0.0.1", 0), backup.ReadOnlyHandler, files, time.time() - 1
        )
        expired_thread = threading.Thread(target=expired.serve_forever, daemon=True)
        expired_thread.start()
        host, port = expired.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/" + name)
        assert conn.getresponse().status == 404
        conn.close()
        expired.shutdown()
        expired_thread.join(timeout=5)
        expired.server_close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_base_url_and_server_manifest_reject_untrusted_routes(tmp_path):
    assert backup._validate_base_url("https://run123.trycloudflare.com")
    for url in (
        "http://run123.trycloudflare.com",
        "https://trycloudflare.com",
        "https://a.b.trycloudflare.com",
        "https://run123.trycloudflare.com/path",
        "https://user@run123.trycloudflare.com",
        "https://run123.trycloudflare.com:443",
        "https://run123.trycloudflare.com?x=1",
        "https://run123.trycloudflare.com.evil.example",
    ):
        try:
            backup._validate_base_url(url)
        except backup.BackupError:
            pass
        else:
            raise AssertionError(f"accepted invalid base URL: {url}")

    data = b"asset"
    digest = hashlib.sha256(data).hexdigest()
    name = f"sha256-{digest}.tar.gz"
    path = tmp_path / name
    path.write_bytes(data)
    path.chmod(0o600)
    manifest = {
        "schema": "harbichess-a100-release-package-v1",
        "archives": [{"name": name, "path": name, "bytes": len(data), "sha256": "0" * 64}],
    }
    (tmp_path / "package.json").write_text(json.dumps(manifest))
    try:
        backup.build_server_map(tmp_path)
    except backup.BackupError as exc:
        assert "sha-name-or-content" in str(exc)
    else:
        raise AssertionError("server accepted wrong allowlist digest")
