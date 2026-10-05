import base64
import copy
import io
import time
import zipfile

import artifact_capsule as c
import pytest


def fixture():
    originals = {
        f"{s}/{n}": (f"{s}-{n}".encode() * 3)
        for s in (20262805, 20262806)
        for n in ("actions-00008192.json.gz", "actor-config.json")
    }
    data = c.capsule(originals)
    m = dict(
        readiness_sha256="90162446656105007ca713c17b01f4e0a762d694a179a5be815e0ba6174bae51",
        source_commit="6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
        action_pins={
            "checkout": "11bd71901bbe5b1630ceea73d27597364c9af683",
            "upload_artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
        },
        schema="fresh8192-exact-capsule-v2",
        status="ROOT-approved-frozen-transport",
        workflow_path=c.WORKFLOW,
        upstream_commit="a" * 40,
        milestone_actions=8192,
        final=True,
        files=[dict(path=p, bytes=len(d), sha256=c.sha(d)) for p, d in originals.items()],
        receiver_sha256="c" * 64,
        workflow_sha256="d" * 64,
        source_bindings={
            "release_transport_frozen.py": "e" * 64,
            "native_delta_codec.py": "f" * 64,
        },
        capsule_bytes=len(data),
        capsule_sha256=c.sha(data),
        chunks=[dict(ordinal=0, bytes=len(data), sha256=c.sha(data))],
    )
    ctrl = dict(
        schema="cpu-native-delta-transport-control-v1",
        status="ROOT-approved-frozen-transport",
        operator_end_epoch=c.release.END,
        started_epoch=time.time() - 1,
        deadline_epoch=time.time() + 60,
    )
    run = dict(
        repository=dict(full_name=c.release.REPO),
        head_sha=m["upstream_commit"],
        path=c.WORKFLOW,
        event="workflow_dispatch",
        status="completed",
        conclusion="success",
    )
    run["source_blobs"] = c.source_seals(m)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("chunk.bin", data)
    archive = buf.getvalue()
    art = dict(
        name=c.artifact_name(m, 0),
        expired=False,
        workflow_run=dict(id=7),
        size_in_bytes=len(archive),
    )
    return m, data, ctrl, run, art, archive


def test_exact_roundtrip():
    m, data, ctrl, run, art, archive = fixture()
    c.verify(data, m)
    assert c.chunk_input(m, 0, base64.b64encode(data).decode()) == data
    assert c.aggregate(m, [7], lambda *args: (run, [art], archive), ctrl) == data


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "b" * 40),
        ("path", "foreign.yml"),
        ("event", "pull_request"),
        ("conclusion", "failure"),
    ],
)
def test_upstream_binding(field, value):
    m, _, ctrl, run, art, archive = fixture()
    run[field] = value
    with pytest.raises(ValueError, match="upstream-run"):
        c.aggregate(m, [7], lambda *args: (run, [art], archive), ctrl)


@pytest.mark.parametrize("mutation", ["size", "sha", "duplicate", "traversal", "draft", "oversize"])
def test_manifest_rejections(mutation):
    m, data, *_ = fixture()
    m = copy.deepcopy(m)
    if mutation == "size":
        m["files"][0]["bytes"] += 1
    elif mutation == "sha":
        m["files"][0]["sha256"] = "0" * 64
    elif mutation == "duplicate":
        m["files"][0] = m["files"][1]
    elif mutation == "traversal":
        m["files"][0]["path"] = "../secret"
    elif mutation == "draft":
        m["status"] = "DRAFT"
    elif mutation == "oversize":
        m["files"][0]["bytes"] = c.LIMIT
    with pytest.raises(ValueError):
        c.verify(data, m)


def test_corrupt_archive_and_duplicate_run():
    m, _, ctrl, run, art, archive = fixture()
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        c.aggregate(m, [7], lambda *args: (run, [art], archive[:-5]), ctrl)
    with pytest.raises(ValueError, match="duplicate-run"):
        c.aggregate(m, [7, 7], lambda *args: (run, [art], archive), ctrl)


def test_artifact_wrong_binding():
    m, _, ctrl, run, art, archive = fixture()
    art["workflow_run"]["id"] = 8
    with pytest.raises(ValueError, match="artifact-binding"):
        c.aggregate(m, [7], lambda *args: (run, [art], archive), ctrl)


class MockRelease:
    def __init__(self, data):
        self.data, self.rows, self.writes = data, [], 0

    def inventory(self):
        return self.rows

    def upload(self, name, data):
        self.writes += 1
        row = dict(
            id=11,
            name=name,
            size=len(data),
            state="uploaded",
            url=f"{c.release.API}/repos/{c.release.REPO}/releases/assets/11",
            browser_download_url=c.release.public_url(name),
            digest="sha256:" + c.sha(data),
        )
        self.rows.append(row)
        return row

    def download(self, name, size):
        return self.data


def test_idempotency_anonymous_corruption_and_cap():
    t = MockRelease(b"capsule")
    name = "sha256-" + c.sha(t.data) + ".tar.gz"
    assert c.release.publish_verified(t, name, t.data)["anonymous_full_body_sha_verified"]
    assert c.release.publish_verified(t, name, t.data)["disposition"] == "matching-retry-no-write"
    assert t.writes == 1
    t.data = b"corrupt"
    with pytest.raises(c.release.Failure):
        c.release.publish_verified(t, name, b"capsule")
    t = MockRelease(b"a")
    t.rows = [dict(name=f"old{i}", size=1) for i in range(200)]
    with pytest.raises(c.release.Failure, match="caps"):
        c.release.publish_verified(t, "new", b"a")


def test_temporary_inventory_duplicate_retry_and_full_page_bound():
    m, *_ = fixture()
    row = dict(name=c.artifact_name(m, 0), expired=False, size_in_bytes=200)
    names, used = c.temporary_inventory(m, lambda p: {"artifacts": [row]})
    assert names == {row["name"]} and used == 200
    with pytest.raises(ValueError, match="duplicate-temp"):
        c.temporary_inventory(m, lambda p: {"artifacts": [row, row]})
    other = dict(name="other", expired=False, size_in_bytes=200)
    with pytest.raises(ValueError, match="incomplete"):
        c.temporary_inventory(m, lambda p: {"artifacts": [other] * 100})


def test_streamed_output_bound_and_deadline():
    import sys
    from time import time

    assert c.read_process([sys.executable, "-c", "print('ok',end='')"], 2, time() + 2) == b"ok"
    with pytest.raises(ValueError, match="size-bound"):
        c.read_process([sys.executable, "-c", "print('x'*100000)"], 10, time() + 2)
    with pytest.raises(ValueError, match="expired"):
        c.read_process([sys.executable, "-c", "import time; time.sleep(1)"], 10, time() - 1)


def test_authorizer_exact_sha_source_and_anchors():
    import json

    import authorize_manifest as a

    m, *_ = fixture()
    m["producer_bindings"] = [
        dict(
            seed=s,
            actions=8192,
            model_sha256=c.release.codec.E8_SHA256,
            journal_schema="fresh-qsearch-selfplay-journal-v2",
        )
        for s in (20262805, 20262806)
    ]
    raw = json.dumps(m).encode()
    approval = dict(
        schema="fresh8192-root-manifest-approval-v2",
        status="ROOT-reviewed-prospectively-approved",
        manifest_sha256=c.sha(raw),
        source_seals=c.source_seals(m),
        producer_bindings=m["producer_bindings"],
    )
    assert a.authorize(raw, approval) == {c.sha(raw): "manifest-8192-DRAFT.json"}
    with pytest.raises(ValueError, match="approved-manifest"):
        a.authorize(raw + b" ", approval)
    approval["source_seals"] = {}
    with pytest.raises(ValueError, match="source-seals"):
        a.authorize(raw, approval)


@pytest.mark.parametrize("allow", [{}, {"a" * 64: "../secret"}])
def test_empty_or_arbitrary_allowlist_fails_before_file_read(tmp_path, monkeypatch, allow):
    import json

    (tmp_path / "allowlist-PENDING.json").write_text(json.dumps(allow))
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setenv("MANIFEST_SHA", "a" * 64)
    monkeypatch.setenv("MANIFEST_PREFIX", "a" * 12)
    with pytest.raises(ValueError, match="manifest-allowlist|fixed-manifest-path"):
        c.main()


def test_source_blob_mismatch_even_same_commit_rejected():
    m, _, ctrl, run, art, archive = fixture()
    run["source_blobs"] = dict(run["source_blobs"])
    run["source_blobs"][c.WORKFLOW] = "0" * 64
    with pytest.raises(ValueError, match="upstream-run"):
        c.aggregate(m, [7], lambda *args: (run, [art], archive), ctrl)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "fresh4096-exact-capsule-draft-v1"),
        ("milestone_actions", 4096),
        ("final", False),
        ("readiness_sha256", "0" * 64),
        ("source_commit", "b" * 40),
    ],
)
def test_final_only_bindings(field, value):
    m, data, *_ = fixture()
    m[field] = value
    with pytest.raises(ValueError):
        c.verify(data, m)


@pytest.mark.parametrize("body", [b"ab", b"abcd"])
def test_anonymous_public_download_short_or_overlong(body):
    from contextlib import closing

    m, _, ctrl, *_ = fixture()
    transport = c.release.Transport("synthetic-not-credential", ctrl)
    transport.request = lambda url, public: closing(io.BytesIO(body))
    with pytest.raises(c.release.Failure, match="full-body-size"):
        transport.download("fixed.tar.gz", 3)


def test_launcher_payload_limits_and_no_arbitrary_manifest():
    import json

    import launcher_inputs as launch

    m, data, *_ = fixture()
    raw = json.dumps(m).encode()
    digest = c.sha(raw)
    built = json.loads(
        launch.dispatch_inputs(
            raw, digest, "chunk", ordinal=0, payload=base64.b64encode(data).decode()
        )
    )
    assert built["inputs"]["manifest_prefix"] == digest[:12]
    assert len(json.dumps(built)) < 65535
    assert (
        json.loads(launch.dispatch_inputs(raw, digest, "aggregate", run_ids=[7]))["inputs"][
            "payload"
        ]
        == ""
    )
    with pytest.raises(ValueError, match="approved-manifest"):
        launch.dispatch_inputs(raw, "0" * 64, "aggregate", run_ids=[7])
    with pytest.raises(ValueError, match="run-ids"):
        launch.dispatch_inputs(raw, digest, "aggregate", run_ids=[7, 7])
