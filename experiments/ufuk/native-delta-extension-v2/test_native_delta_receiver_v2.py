"""Mocked transport only; no network, upload, workflow dispatch or model execution."""

import copy
import io
import os
import tarfile
import time
from pathlib import Path

import native_delta_binding_v2 as codec
import native_delta_receiver_v2 as receiver
import pytest

PARENT = Path(
    "/workspace/work/harbichess/a100/restoration/local-rehearsal-content/"
    "harbichess-inputs/initial-e8.safetensors"
)
PAYLOAD_ROOT = Path(os.environ.get("HARBICHESS_NATIVE_V2_FIXTURES", str(receiver.ROOT)))


class FakeTransport:
    def __init__(self):
        self.rows, self.bodies, self.writes = [], {}, []
        self.corrupt_name = None
        self.extra_after_write = False

    def inventory(self):
        return copy.deepcopy(self.rows)

    def seed(self, name, data):
        row = {
            "id": len(self.rows) + 1,
            "name": name,
            "size": len(data),
            "state": "uploaded",
            "digest": "sha256:" + codec.digest(data),
            "url": f"{receiver.API}/repos/{receiver.REPO}/releases/assets/{len(self.rows) + 1}",
            "browser_download_url": receiver.public_url(name),
        }
        self.rows.append(row)
        self.bodies[name] = data
        return copy.deepcopy(row)

    def upload(self, name, data):
        self.writes.append(name)
        row = self.seed(name, data)
        if self.extra_after_write:
            self.rows.append(copy.deepcopy(row))
        return row

    def download(self, name, size):
        data = self.bodies[name]
        if name == self.corrupt_name:
            data = bytes([data[0] ^ 1]) + data[1:]
        assert len(data) == size
        return data

    def parent(self):
        return PARENT.read_bytes()


@pytest.fixture(scope="module")
def fixture():
    manifest = receiver.load_manifest(next(iter(receiver.ALLOWLIST)))
    chunks = [
        (PAYLOAD_ROOT / "20262605-shrunk" / f"chunk-{index:03d}.base64").read_text()
        for index in range(len(manifest["chunks"]))
    ]
    codec.reassemble(manifest, chunks)
    return manifest, chunks


def inputs(manifest, chunks, ordinal=0):
    return {
        "packet_sha256": manifest["packet_sha256"],
        "ordinal": str(ordinal),
        "count": str(len(chunks)),
        "chunk_sha256": manifest["chunks"][ordinal]["sha256"],
        "payload": chunks[ordinal],
    }


def test_idempotent_matching_retry_anonymous_bytes_checked_and_no_second_write(fixture):
    manifest, chunks = fixture
    transport = FakeTransport()
    first = receiver.receive_chunk(manifest, inputs(manifest, chunks), transport)
    second = receiver.receive_chunk(manifest, inputs(manifest, chunks), transport)
    assert first["disposition"] == "new-upload"
    assert second["disposition"] == "matching-retry-no-write"
    assert len(transport.writes) == 1
    assert second["anonymous_full_body_sha_verified"] is True


def test_existing_mismatched_name_never_overwritten(fixture):
    manifest, chunks = fixture
    transport = FakeTransport()
    name = receiver.chunk_name(manifest, 0)
    transport.seed(name, b"wrong")
    before = copy.deepcopy(transport.rows)
    with pytest.raises(receiver.Failure, match="metadata"):
        receiver.receive_chunk(manifest, inputs(manifest, chunks), transport)
    assert transport.writes == [] and transport.rows == before


def test_accepted_upload_corrupt_readback_fails_without_completion(fixture):
    manifest, chunks = fixture
    transport = FakeTransport()
    transport.corrupt_name = receiver.chunk_name(manifest, 0)
    with pytest.raises(receiver.Failure, match="anonymous-full-body-sha"):
        receiver.receive_chunk(manifest, inputs(manifest, chunks), transport)
    assert len(transport.writes) == 1  # Accepted partial is preserved; no clobber/retry reset.


@pytest.mark.parametrize("kind", ["wrong-count", "noncanonical-ordinal", "changed-body"])
def test_invalid_dispatch_inputs_fail_before_any_upload(fixture, kind):
    manifest, chunks = fixture
    data = inputs(manifest, chunks)
    if kind == "wrong-count":
        data["count"] = "1"
    elif kind == "noncanonical-ordinal":
        data["ordinal"] = "00"
    else:
        data["payload"] = "A" + data["payload"][1:]
    transport = FakeTransport()
    with pytest.raises(receiver.Failure):
        receiver.receive_chunk(manifest, data, transport)
    assert not transport.writes


@pytest.mark.parametrize("limit", ["count", "bytes"])
def test_unchanged_aggregate_caps_fail_before_new_write(limit):
    transport = FakeTransport()
    if limit == "count":
        transport.rows = [{"size": 0} for _ in range(receiver.ASSET_CAP)]
    else:
        transport.rows = [{"size": receiver.BYTE_CAP}]
    for index, row in enumerate(transport.rows):
        row["name"] = f"old{index}"
    with pytest.raises(receiver.Failure, match="aggregate-caps"):
        receiver.publish_verified(transport, "new", b"x")
    assert not transport.writes


def test_duplicate_post_upload_publication_is_not_success(fixture):
    manifest, chunks = fixture
    transport = FakeTransport()
    transport.extra_after_write = True
    with pytest.raises(receiver.Failure, match="not-uniquely-published"):
        receiver.receive_chunk(manifest, inputs(manifest, chunks), transport)


def test_real_value_aggregate_preserves_all_original_files_and_retry_no_write(fixture):
    manifest, chunks = fixture
    transport = FakeTransport()
    for index, chunk in enumerate(chunks):
        transport.seed(receiver.chunk_name(manifest, index), chunk.encode("ascii"))
    result = receiver.aggregate(manifest, transport)
    assert result["status"] == "public-exact-bytes-pass"
    assert result["native_strict_runtime_load_executed"] is False
    assert len(transport.writes) == 1
    receiver.verify_capsule(transport.bodies[result["capsule"]["name"]], manifest["files"])
    repeated = receiver.aggregate(manifest, transport)
    assert repeated["capsule"]["sha256"] == result["capsule"]["sha256"]
    assert repeated["capsule"]["disposition"] == "matching-retry-no-write"
    assert len(transport.writes) == 1


def test_aggregate_rejects_missing_or_corrupt_chunk_before_capsule_upload(fixture):
    manifest, chunks = fixture
    transport = FakeTransport()
    with pytest.raises(receiver.Failure, match="chunk-missing"):
        receiver.aggregate(manifest, transport)
    for index, chunk in enumerate(chunks):
        transport.seed(receiver.chunk_name(manifest, index), chunk.encode("ascii"))
    transport.corrupt_name = receiver.chunk_name(manifest, 1)
    with pytest.raises(receiver.Failure, match="chunk-full-sha"):
        receiver.aggregate(manifest, transport)
    assert not transport.writes


def test_capsule_is_deterministic_regular_exact_three_and_rejects_link():
    originals = {name: name.encode("ascii") for name in codec.FILES}
    expected = {
        name: {"bytes": len(data), "sha256": codec.digest(data)} for name, data in originals.items()
    }
    assert receiver.deterministic_capsule(originals) == receiver.deterministic_capsule(originals)
    receiver.verify_capsule(receiver.deterministic_capsule(originals), expected)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as archive:
        info = tarfile.TarInfo(codec.FILES[0])
        info.type, info.linkname = tarfile.LNKTYPE, "arbitrary"
        archive.addfile(info)
    with pytest.raises(receiver.Failure, match="extra-link"):
        receiver.verify_capsule(raw.getvalue(), expected)


def test_canonical_full_page_no_link_queries_next_page_not_declared_complete(monkeypatch):
    control = {
        "schema": "cpu-native-delta-transport-control-v2",
        "status": "ROOT-approved-frozen-transport",
        "operator_end_epoch": receiver.END,
        "started_epoch": time.time() - 10,
        "deadline_epoch": time.time() + 10,
    }
    transport = receiver.Transport("FAKE-not-a-credential", control)
    calls = []

    def fake(path, data=None):
        calls.append(path)
        if path.endswith(str(receiver.RELEASE)):
            return {
                "id": receiver.RELEASE,
                "tag_name": receiver.TAG,
                "draft": False,
                "upload_url": (
                    f"{receiver.UPLOAD}/repos/{receiver.REPO}"
                    f"/releases/{receiver.RELEASE}/assets{{?name,label}}"
                ),
            }
        if path.endswith("page=1"):
            return [{"id": index + 1, "name": f"old{index}", "size": 0} for index in range(100)]
        assert path.endswith("page=2")
        return []

    monkeypatch.setattr(transport, "api_json", fake)
    assert len(transport.inventory()) == 100
    assert calls[-1].endswith("page=2")


def test_clock_refuses_pending_or_extended_expired_and_operator_decode():
    assert receiver.END == 1791273600
    import datetime

    assert (
        datetime.datetime.fromtimestamp(receiver.END, datetime.UTC).isoformat()
        == "2026-10-06T08:00:00+00:00"
    )
    control = {
        "schema": "cpu-native-delta-transport-control-v2",
        "status": "ROOT-approved-frozen-transport",
        "operator_end_epoch": receiver.END,
        "started_epoch": receiver.END - 1000,
        "deadline_epoch": receiver.END,
    }
    assert receiver.check_clock(control, receiver.END - 1) == 1
    with pytest.raises(receiver.Failure):
        receiver.check_clock(control, receiver.END)
    control["deadline_epoch"] += 1
    with pytest.raises(receiver.Failure):
        receiver.check_clock(control, receiver.END - 1)
    control["status"] = "PENDING"
    with pytest.raises(receiver.Failure):
        receiver.check_clock(control, receiver.END - 1)


def test_parent_stream_verifies_entire_archive_and_original_e8_member(monkeypatch):
    # Small synthetic container with the REAL e8 bytes; no public network or fake public claim.
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        body = PARENT.read_bytes()
        info = tarfile.TarInfo(codec.PUBLIC_PARENT_MEMBER)
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    archive = raw.getvalue()
    monkeypatch.setattr(codec, "PUBLIC_PARENT_ARCHIVE_BYTES", len(archive))
    monkeypatch.setattr(codec, "PUBLIC_PARENT_ARCHIVE_SHA256", codec.digest(archive))
    monkeypatch.setattr(receiver, "PARENT_NAME", "sha256-" + codec.digest(archive) + ".tar.gz")
    control = {
        "schema": "cpu-native-delta-transport-control-v2",
        "status": "ROOT-approved-frozen-transport",
        "operator_end_epoch": receiver.END,
        "started_epoch": time.time() - 1,
        "deadline_epoch": time.time() + 10,
    }
    transport = receiver.Transport("FAKE", control)
    fake = FakeTransport()
    fake.seed(receiver.PARENT_NAME, archive)
    monkeypatch.setattr(transport, "inventory", fake.inventory)
    monkeypatch.setattr(transport, "request", lambda url, public: io.BytesIO(archive))
    assert transport.parent() == body
    changed = archive[:-1] + bytes([archive[-1] ^ 1])
    monkeypatch.setattr(transport, "request", lambda url, public: io.BytesIO(changed))
    with pytest.raises((receiver.Failure, OSError, EOFError)):
        transport.parent()


def test_public_download_and_redirect_never_forward_authorization(monkeypatch):
    control = {
        "schema": "cpu-native-delta-transport-control-v2",
        "status": "ROOT-approved-frozen-transport",
        "operator_end_epoch": receiver.END,
        "started_epoch": time.time() - 1,
        "deadline_epoch": time.time() + 10,
    }
    transport = receiver.Transport("FAKE", control)

    class Capture:
        def open(self, request, timeout):
            assert request.method == "GET"
            assert not request.has_header("Authorization")
            assert request.full_url == receiver.public_url("allowed")
            return io.BytesIO(b"abc")

    monkeypatch.setattr(transport, "public_opener", Capture())
    assert transport.download("allowed", 3) == b"abc"
    original = receiver.urllib.request.Request(
        receiver.public_url("allowed"), headers={"Authorization": "FAKE"}
    )
    handler = receiver.PublicRedirect()
    redirected = handler.redirect_request(
        original, None, 302, "", {}, "https://release-assets.githubusercontent.com/allowed"
    )
    assert not redirected.has_header("Authorization")
    with pytest.raises(receiver.Failure):
        handler.redirect_request(original, None, 302, "", {}, "https://foreign.example/asset")
