"""FINAL8192 v3 direct Release recovery; no artifact/Azure request code."""

import base64
import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath

import release_transport_frozen as release

ROOT = Path(__file__).resolve().parent
LIMIT = 4 * 1024**2
TEMP_LIMIT = 8 * 1024**2
CHUNK = 43500  # 58000 base64 characters
PUBLICATION_DIR = "experiments/ufuk/fresh-final8192-release-capsule-v3"
WORKFLOW = ".github/workflows/fresh-final8192-release-capsule-v3.yml"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, code):
    if not ok:
        raise ValueError(code)


def validate(m):
    require(
        m["readiness_sha256"] == "90162446656105007ca713c17b01f4e0a762d694a179a5be815e0ba6174bae51",
        "final-readiness-binding",
    )
    require(
        m["source_commit"] == "6fcc8b476d25495d1c9c413e55b2c7ba4794013e", "actor-source-binding"
    )
    require(
        m["action_pins"]
        == {
            "checkout": "11bd71901bbe5b1630ceea73d27597364c9af683",
        },
        "action-pins",
    )
    require(m["schema"] == "fresh8192-exact-release-capsule-v3", "schema")
    require(m["status"] == "ROOT-approved-frozen-transport", "draft-not-approved")
    require(
        m["workflow_path"] == WORKFLOW
        and (m["upstream_commit"] is None or len(m["upstream_commit"]) == 40),
        "upstream",
    )
    require(m["milestone_actions"] == 8192 and m["final"] is True, "not-final")
    require(
        set(m["source_bindings"]) == {"release_transport_frozen.py", "native_delta_codec.py"},
        "fixed-source-paths",
    )
    for digest in [m["receiver_sha256"], m["workflow_sha256"], *m["source_bindings"].values()]:
        require(
            len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "source-sha-format"
        )
    expected = {
        f"{seed}/{name}"
        for seed in (20262805, 20262806)
        for name in ("actions-00008192.json.gz", "actor-config.json")
    }
    rows = m["files"]
    require(len(rows) == 4 and {r["path"] for r in rows} == expected, "fixed-four-paths")
    for r in rows:
        p = PurePosixPath(r["path"])
        require(not p.is_absolute() and ".." not in p.parts, "path")
        require(type(r["bytes"]) is int and 0 < r["bytes"] <= LIMIT // 2, "file-size")
        require(len(r["sha256"]) == 64, "file-sha")
    require(sum(r["bytes"] for r in rows) <= LIMIT, "total-size")
    require(0 < m["capsule_bytes"] <= LIMIT and len(m["capsule_sha256"]) == 64, "capsule-bound")
    chunks = m["chunks"]
    require(len(chunks) == (m["capsule_bytes"] + CHUNK - 1) // CHUNK, "chunk-count")
    require([r["ordinal"] for r in chunks] == list(range(len(chunks))), "ordinal")
    require(sum(r["bytes"] for r in chunks) == m["capsule_bytes"], "chunk-size-sum")
    for r in chunks:
        require(0 < r["bytes"] <= CHUNK and len(r["sha256"]) == 64, "chunk-bound")


def capsule(originals):
    out = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=out, mode="wb", mtime=0, filename="") as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar,
    ):
        for path, data in sorted(originals.items()):
            info = tarfile.TarInfo(path)
            info.size, info.mode, info.mtime = len(data), 0o600, 0
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def verify(data, m):
    validate(m)
    require(len(data) == m["capsule_bytes"] and sha(data) == m["capsule_sha256"], "capsule-sha")
    expected = {r["path"]: r for r in m["files"]}
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r|gz") as tar:
        for member in tar:
            require(
                member.isreg() and member.name in expected and member.name not in seen, "member"
            )
            row = expected[member.name]
            require(member.size == row["bytes"], "member-size")
            body = tar.extractfile(member).read(row["bytes"] + 1)
            require(len(body) == row["bytes"] and sha(body) == row["sha256"], "member-sha")
            seen.add(member.name)
    require(seen == set(expected), "missing-member")


def artifact_name(m, ordinal):
    return f"fresh8192-{m['capsule_sha256']}-{ordinal:03d}"


def chunk_input(m, ordinal, payload):
    validate(m)
    require(type(ordinal) is int and 0 <= ordinal < len(m["chunks"]), "ordinal")
    require(len(payload) <= 58000, "dispatch-bound")
    data = base64.b64decode(payload, validate=True)
    r = m["chunks"][ordinal]
    require(len(data) == r["bytes"] and sha(data) == r["sha256"], "chunk-sha")
    return data


def chunk_name(m, ordinal):
    return f"fresh8192v3-{m['capsule_sha256']}-chunk-{ordinal:03d}.bin"


def aggregate_release(m, transport, control):
    validate(m)
    inventory = transport.inventory()
    parts = []
    total = 0
    for row in m["chunks"]:
        release.check_clock(control)
        ordinal = row["ordinal"]
        name = chunk_name(m, ordinal)
        matches = [a for a in inventory if a["name"] == name]
        require(len(matches) == 1, "public-chunk-missing-or-duplicate")
        release.check_asset(matches[0], name, row["bytes"], row["sha256"])
        raw = transport.download(name, row["bytes"])
        require(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "public-chunk-sha")
        total += len(raw)
        require(total <= LIMIT and total <= TEMP_LIMIT, "public-aggregate-memory-bound")
        parts.append(raw)
    data = b"".join(parts)
    verify(data, m)
    return data


def source_seals(m):
    return {
        WORKFLOW: m["workflow_sha256"],
        f"{PUBLICATION_DIR}/artifact_capsule.py": m["receiver_sha256"],
        **{f"{PUBLICATION_DIR}/{k}": v for k, v in m["source_bindings"].items()},
    }


def main():
    # Fixed local manifest SHA allowlist. ROOT must freeze this file before dispatch.
    allow = json.loads((ROOT / "allowlist-PENDING.json").read_bytes())
    digest = os.environ["MANIFEST_SHA"]
    require(os.environ.get("MANIFEST_PREFIX") == digest[:12], "manifest-run-name-prefix")
    require(digest in allow, "manifest-allowlist")
    require(allow[digest] == "manifest-8192-DRAFT.json", "fixed-manifest-path")
    raw = (ROOT / allow[digest]).read_bytes()
    require(sha(raw) == digest, "manifest-sha")
    m = json.loads(raw)
    require(sha(Path(__file__).read_bytes()) == m["receiver_sha256"], "receiver-binding")
    for name, support_digest in m["source_bindings"].items():
        require(
            name in ("release_transport_frozen.py", "native_delta_codec.py")
            and sha((ROOT / name).read_bytes()) == support_digest,
            "support-source-binding",
        )
    require(
        set(m["source_bindings"]) == {"release_transport_frozen.py", "native_delta_codec.py"},
        "support-source-set",
    )
    require(
        sha((ROOT.parents[2] / WORKFLOW).read_bytes()) == m["workflow_sha256"],
        "checked-out-workflow-sha",
    )
    control = json.loads((ROOT / "transport-control-PENDING.json").read_bytes())
    release.check_clock(control)
    # Frozen controls retain the FAILEDv2 original clock; recovery does not restart it.
    require(
        control["started_epoch"] == 1791226974.7925532
        and control["deadline_epoch"] == 1791228774.7925532,
        "original-v2-clock-required",
    )
    transport = release.Transport(os.environ["GH_TOKEN"], control)
    if os.environ["MODE"] == "chunk":
        ordinal = int(os.environ["ORDINAL"])
        data = chunk_input(m, ordinal, os.environ["PAYLOAD"])
        proof = release.publish_verified(transport, chunk_name(m, ordinal), data)
    else:
        require(os.environ["MODE"] == "aggregate" and os.environ.get("PAYLOAD", "") == "", "mode")
        data = aggregate_release(m, transport, control)
        proof = release.publish_verified(transport, f"sha256-{sha(data)}.tar.gz", data)
        verify(transport.download(proof["name"], len(data)), m)
    release.check_clock(control)
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
