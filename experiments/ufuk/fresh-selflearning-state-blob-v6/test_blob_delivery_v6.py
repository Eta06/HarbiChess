"""Synthetic isolated API responses and upload mocks; no actual network or Git."""

import base64
import json
import time

import blob_delivery_v6 as delivery
import pytest
import receiver_v6 as receiver
import release_transport_v6 as release
import state_bundle as bundle


@pytest.fixture
def blob(monkeypatch):
    raw = b"fixed-local-own-chess-native-bytes"
    monkeypatch.setattr(delivery, "CAPSULE_BYTES", len(raw))
    monkeypatch.setattr(delivery, "BLOB_SHA1", delivery.git_blob_sha(raw))
    monkeypatch.setattr(delivery, "CAPSULE_SHA", bundle.sha(raw))
    m = dict(
        transport_schema=delivery.TRANSPORT_SCHEMA,
        capsule_bytes=len(raw),
        capsule_sha256=bundle.sha(raw),
        blob=dict(
            sha1=delivery.git_blob_sha(raw),
            bytes=len(raw),
            sha256=bundle.sha(raw),
            repository=release.REPO,
            status="ROOT-verified-unreferenced-transient-blob",
        ),
    )
    response = dict(
        sha=m["blob"]["sha1"],
        size=len(raw),
        encoding="base64",
        url=release.API + delivery.BLOB_API + "/" + m["blob"]["sha1"],
        content=base64.encodebytes(raw).decode(),
    )
    return raw, m, response


def test_git_sha1_hashes_blob_header_not_raw_file():
    assert delivery.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert delivery.git_blob_sha(b"hello") != delivery.git_blob_sha(b"hello\n")


def test_pure_post_json_and_bounded_wrapped_get_roundtrip(blob):
    raw, m, response = blob
    endpoint, body = delivery.create_blob_request(raw, m)
    assert endpoint == "/repos/Eta06/HarbiChess/git/blobs"
    assert base64.b64decode(json.loads(body)["content"]) == raw
    assert delivery.verify_create_response(response, m) == m["blob"]["sha1"]
    assert delivery.decode_blob_response(json.dumps(response).encode(), m) == raw


@pytest.mark.parametrize(
    "mutation", ["repo", "sha1", "size", "encoding", "content", "url", "extra-content"]
)
def test_corruption_wrong_repository_and_blob_rejected(blob, mutation):
    _, m, response = blob
    if mutation == "repo":
        m["blob"]["repository"] = "foreign/repository"
    elif mutation == "sha1":
        response["sha"] = "0" * 40
    elif mutation == "size":
        response["size"] += 1
    elif mutation == "encoding":
        response["encoding"] = "utf-8"
    elif mutation == "content":
        response["content"] = base64.b64encode(b"x" * m["blob"]["bytes"]).decode()
    elif mutation == "url":
        response["url"] = "https://api.github.com/repos/foreign/repository/git/blobs/x"
    else:
        response["content"] += "A"
    with pytest.raises(ValueError):
        delivery.decode_blob_response(json.dumps(response).encode(), m)


def test_pending_blob_rejected_and_unknown_create_response_not_confirmed(blob):
    _, m, response = blob
    m["blob"]["status"] = "PENDING-ROOT-POST-VERIFY"
    with pytest.raises(ValueError):
        delivery.decode_blob_response(json.dumps(response).encode(), m)
    with pytest.raises(ValueError):
        delivery.verify_create_response({"sha": "unknown"}, m)


def test_only_fixed_local_capsule_can_prepare_post(blob):
    raw, m, _ = blob
    with pytest.raises(ValueError):
        delivery.create_blob_request(raw + b"x", m)
    m["blob"]["sha1"] = "0" * 40
    with pytest.raises(ValueError):
        delivery.create_blob_request(raw, m)


def test_fetch_exact_repository_sha_bound_not_response_url(blob):
    raw, m, response = blob
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            result = json.dumps(response).encode()
            assert len(result) < size
            return result

    class Transport:
        def request(self, url):
            calls.append(url)
            return Response()

    assert delivery.fetch_verified_blob(Transport(), m) == raw
    assert calls == [release.API + delivery.BLOB_API + "/" + m["blob"]["sha1"]]


def control():
    now = time.time()
    return dict(
        schema="fresh-selflearning-state-blob-transport-control-v6",
        status="ROOT-approved-frozen-transport",
        started_epoch=now - 1,
        deadline_epoch=min(now + 100, release.END),
        operator_end_epoch=release.END,
    )


def test_one_capsule_upload_and_idempotent_second_call(blob, monkeypatch):
    raw, m, _ = blob
    m["release"] = {"id": 123}
    events = []

    def decode(data, parent, manifest):
        assert data == raw and parent == b"exact-parent"
        events.append("decode22")
        return {p: p.encode() for p in bundle.PATHS}

    monkeypatch.setattr(bundle, "decode", decode)

    class Transport:
        control = control()

        def __init__(self):
            self.rows = []
            self.uploads = 0

        def inventory(self):
            return self.rows

        def asset(self, row, name, data):
            assert row["name"] == name and data == raw

        def upload(self, name, data):
            assert events[-1] == "decode22"
            self.uploads += 1
            events.append("upload")
            self.rows = [dict(id=7, name=name, size=len(data))]
            return self.rows[0]

        def download(self, name, size):
            events.append("anonymous-readback")
            return raw

    t = Transport()
    proof = receiver.publish_capsule(m, t, b"exact-parent", raw)
    assert proof["all22_original_byte_sha_verified"] and len(proof["originals"]) == 22
    assert events == ["decode22", "upload", "anonymous-readback", "anonymous-readback", "decode22"]
    receiver.publish_capsule(m, t, b"exact-parent", raw)
    assert t.uploads == 1


def test_preupload_all22_failure_writes_nothing(blob, monkeypatch):
    raw, m, _ = blob

    def decode(*args):
        raise ValueError("raw-native-contract-corruption")

    monkeypatch.setattr(bundle, "decode", decode)

    class Transport:
        def upload(self, *args):
            raise AssertionError("must-not-write")

    with pytest.raises(ValueError, match="raw-native-contract"):
        receiver.publish_capsule(m, Transport(), b"parent", raw)


def test_old_control_and_expired_new_control_never_pass():
    c = control()
    c["schema"] = "fresh-selflearning-state-transport-control-v5"
    with pytest.raises(release.Failure):
        release.check_clock(c)
    c = control()
    with pytest.raises(release.Failure):
        release.check_clock(c, c["deadline_epoch"])
