import base64
import copy
import io
import json
import time

import artifact_capsule as c
import pytest


def fixture():
    originals = {
        f"{s}/{n}": f"{s}-{n}".encode() * 3
        for s in (20262805, 20262806)
        for n in ("actions-00008192.json.gz", "actor-config.json")
    }
    raw = c.capsule(originals)
    m = dict(
        schema="fresh8192-exact-release-capsule-v3",
        status="ROOT-approved-frozen-transport",
        workflow_path=c.WORKFLOW,
        upstream_commit=None,
        final=True,
        milestone_actions=8192,
        readiness_sha256="90162446656105007ca713c17b01f4e0a762d694a179a5be815e0ba6174bae51",
        source_commit="6fcc8b476d25495d1c9c413e55b2c7ba4794013e",
        action_pins={"checkout": "11bd71901bbe5b1630ceea73d27597364c9af683"},
        receiver_sha256="c" * 64,
        workflow_sha256="d" * 64,
        source_bindings={
            "release_transport_frozen.py": "e" * 64,
            "native_delta_codec.py": "f" * 64,
        },
        files=[dict(path=p, bytes=len(d), sha256=c.sha(d)) for p, d in originals.items()],
        capsule_bytes=len(raw),
        capsule_sha256=c.sha(raw),
        chunks=[dict(ordinal=0, bytes=len(raw), sha256=c.sha(raw))],
    )
    ctrl = dict(
        schema="cpu-native-delta-transport-control-v1",
        status="ROOT-approved-frozen-transport",
        operator_end_epoch=c.release.END,
        started_epoch=time.time() - 1,
        deadline_epoch=time.time() + 60,
    )
    return m, raw, ctrl


class MockRelease:
    def __init__(self):
        self.rows = []
        self.bodies = {}
        self.writes = 0
        self.reads = []

    def inventory(self):
        return self.rows

    def upload(self, name, data):
        self.writes += 1
        ident = len(self.rows) + 1
        row = dict(
            id=ident,
            name=name,
            size=len(data),
            state="uploaded",
            digest="sha256:" + c.sha(data),
            url=f"{c.release.API}/repos/{c.release.REPO}/releases/assets/{ident}",
            browser_download_url=c.release.public_url(name),
        )
        self.rows.append(row)
        self.bodies[name] = data
        return row

    def download(self, name, size):
        self.reads.append((name, size))
        return self.bodies[name]


def test_exact_release_chunks_then_capsule_idempotent_no_azure():
    m, raw, ctrl = fixture()
    t = MockRelease()
    c.release.publish_verified(t, c.chunk_name(m, 0), raw)
    assert c.aggregate_release(m, t, ctrl) == raw
    name = f"sha256-{c.sha(raw)}.tar.gz"
    c.release.publish_verified(t, name, raw)
    assert c.release.publish_verified(t, name, raw)["disposition"] == "matching-retry-no-write"
    assert t.writes == 2
    assert all("blob.core.windows.net" not in n for n, _ in t.reads)
    c.verify(t.download(name, len(raw)), m)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "fresh8192-exact-capsule-v2"),
        ("final", False),
        ("milestone_actions", 4096),
        ("readiness_sha256", "0" * 64),
        ("source_commit", "b" * 40),
        ("capsule_bytes", 5 * 1024**2),
    ],
)
def test_version_scope_source_caps_rejected(field, value):
    m, raw, _ = fixture()
    m[field] = value
    with pytest.raises(ValueError):
        c.verify(raw, m)


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "size", "body", "metadata", "expired"]
)
def test_aggregate_failclosed(mutation):
    m, raw, ctrl = fixture()
    t = MockRelease()
    name = c.chunk_name(m, 0)
    c.release.publish_verified(t, name, raw)
    if mutation == "missing":
        t.rows = []
    elif mutation == "duplicate":
        t.rows *= 2
    elif mutation == "size":
        t.rows[0]["size"] += 1
    elif mutation == "body":
        t.bodies[name] = bytes([raw[0] ^ 1]) + raw[1:]
    elif mutation == "metadata":
        t.rows[0]["browser_download_url"] = "https://foreign.invalid/x"
    else:
        ctrl["deadline_epoch"] = time.time() - 1
    with pytest.raises((ValueError, c.release.Failure)):
        c.aggregate_release(m, t, ctrl)


@pytest.mark.parametrize("mutation", ["path", "sha", "size", "duplicate"])
def test_originalfile_sha_path_size(mutation):
    m, raw, _ = fixture()
    m = copy.deepcopy(m)
    if mutation == "path":
        m["files"][0]["path"] = "../private"
    elif mutation == "sha":
        m["files"][0]["sha256"] = "0" * 64
    elif mutation == "size":
        m["files"][0]["bytes"] += 1
    else:
        m["files"][0] = m["files"][1]
    with pytest.raises(ValueError):
        c.verify(raw, m)


def test_no_new_cap_space_and_public_readback_corruption():
    m, raw, _ = fixture()
    t = MockRelease()
    t.rows = [dict(name=f"old{i}", size=1) for i in range(200)]
    with pytest.raises(c.release.Failure, match="caps"):
        c.release.publish_verified(t, c.chunk_name(m, 0), raw)
    t = MockRelease()
    t.upload("same", raw)
    t.bodies["same"] = b"bad"
    with pytest.raises(c.release.Failure):
        c.release.publish_verified(t, "same", raw)


@pytest.mark.parametrize("body", [b"ab", b"abcd"])
def test_public_download_exact_bound(body):
    from contextlib import closing

    _, _, ctrl = fixture()
    t = c.release.Transport("synthetic-not-credential", ctrl)
    t.request = lambda url, public: closing(io.BytesIO(body))
    with pytest.raises(c.release.Failure, match="full-body-size"):
        t.download("fixed", 3)


def test_launcher_envelope_no_artifact_run_ids():
    import launcher_inputs as launch

    m, raw, _ = fixture()
    manifest = json.dumps(m).encode()
    sha = c.sha(manifest)
    built = json.loads(
        launch.dispatch_inputs(
            manifest, sha, "chunk", ordinal=0, payload=base64.b64encode(raw).decode()
        )
    )
    assert built["inputs"]["manifest_prefix"] == sha[:12] and "run_ids" not in built["inputs"]
    assert len(json.dumps(built)) < 65535
    with pytest.raises(ValueError):
        launch.dispatch_inputs(manifest, sha, "aggregate", run_ids=[7])


@pytest.mark.parametrize("allow", [{}, {"a" * 64: "../secret"}])
def test_empty_or_arbitrary_allowlist_rejected(tmp_path, monkeypatch, allow):
    (tmp_path / "allowlist-PENDING.json").write_text(json.dumps(allow))
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setenv("MANIFEST_SHA", "a" * 64)
    monkeypatch.setenv("MANIFEST_PREFIX", "a" * 12)
    with pytest.raises(ValueError):
        c.main()


def test_authorizer_still_seals_exact_manifest():
    import authorize_manifest as author

    m, _, _ = fixture()
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
        schema="fresh8192-root-manifest-approval-v3",
        status="ROOT-reviewed-prospectively-approved",
        manifest_sha256=c.sha(raw),
        source_seals=c.source_seals(m),
        producer_bindings=m["producer_bindings"],
    )
    assert author.authorize(raw, approval) == {c.sha(raw): "manifest-8192-DRAFT.json"}
    with pytest.raises(ValueError):
        author.authorize(raw + b" ", approval)
