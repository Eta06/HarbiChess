"""Exact fixed-manifest transport; temporary opaque chunks never enter Git."""

import base64
import gzip
import hashlib
import io
import json
import os
import selectors
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import release_transport_frozen as release

ROOT = Path(__file__).resolve().parent
LIMIT = 4 * 1024**2
TEMP_LIMIT = 8 * 1024**2
CHUNK = 43500  # 58000 base64 characters
PUBLICATION_DIR = "experiments/ufuk/fresh-final8192-artifact-capsule-v2"
ACTIVE_CONTROL = None
WORKFLOW = ".github/workflows/fresh-final8192-artifact-capsule-v2.yml"


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
            "upload_artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
        },
        "action-pins",
    )
    require(m["schema"] == "fresh8192-exact-capsule-v2", "schema")
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


def aggregate(m, run_ids, fetch, control):
    validate(m)
    require(len(run_ids) == len(m["chunks"]) and len(set(run_ids)) == len(run_ids), "duplicate-run")
    parts, total = [], 0
    for ordinal, run_id in enumerate(run_ids):
        release.check_clock(control)
        require(type(run_id) is int and run_id > 0, "run-id")
        run, artifacts, archive = fetch(run_id, artifact_name(m, ordinal))
        require(
            run["repository"]["full_name"] == release.REPO
            and len(run["head_sha"]) == 40
            and (m["upstream_commit"] is None or run["head_sha"] == m["upstream_commit"])
            and run["source_blobs"] == source_seals(m)
            and run["path"] == WORKFLOW
            and run["event"] == "workflow_dispatch"
            and run["status"] == "completed"
            and run["conclusion"] == "success",
            "upstream-run",
        )
        require(len(artifacts) == 1, "artifact-count")
        a = artifacts[0]
        require(
            a["name"] == artifact_name(m, ordinal)
            and a["expired"] is False
            and a["workflow_run"]["id"] == run_id
            and type(a["size_in_bytes"]) is int
            and 0 < a["size_in_bytes"] <= 100000,
            "artifact-binding",
        )
        total += len(archive)
        require(total <= TEMP_LIMIT and len(archive) == a["size_in_bytes"], "temporary-quota-size")
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            entries = z.infolist()
            require(
                len(entries) == 1
                and entries[0].filename == "chunk.bin"
                and entries[0].file_size == m["chunks"][ordinal]["bytes"]
                and entries[0].external_attr >> 16 & 0o170000 in (0, 0o100000),
                "zip-member",
            )
            part = z.read(entries[0])
        require(sha(part) == m["chunks"][ordinal]["sha256"], "artifact-chunk-sha")
        parts.append(part)
    data = b"".join(parts)
    verify(data, m)
    return data


def source_seals(m):
    return {
        WORKFLOW: m["workflow_sha256"],
        f"{PUBLICATION_DIR}/artifact_capsule.py": m["receiver_sha256"],
        **{f"{PUBLICATION_DIR}/{k}": v for k, v in m["source_bindings"].items()},
    }


def read_process(command, limit, deadline):
    """Bound output WHILE reading, and terminate an overrun within original clock."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output = bytearray()
    try:
        with selectors.DefaultSelector() as poll:
            poll.register(process.stdout, selectors.EVENT_READ)
            while poll.get_map():
                require(time.time() < deadline, "api-clock-expired")
                for key, _ in poll.select(min(0.1, max(0, deadline - time.time()))):
                    part = os.read(key.fileobj.fileno(), min(65536, limit + 1 - len(output)))
                    require(len(output) + len(part) <= limit, "api-stream-size-bound")
                    if part:
                        output.extend(part)
                    else:
                        poll.unregister(key.fileobj)
            require(process.wait(timeout=max(0.01, deadline - time.time())) == 0, "api-failed")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        process.stdout.close()
    return bytes(output)


def gh(path, binary=False):
    prefix = f"repos/{release.REPO}/"
    require(
        path.startswith(prefix + "actions/") or path.startswith(prefix + "contents/"), "fixed-api"
    )
    require(ACTIVE_CONTROL is not None, "api-requires-original-clock")
    timeout = release.check_clock(ACTIVE_CONTROL)
    raw = read_process(["gh", "api", path], TEMP_LIMIT, time.time() + timeout)
    return raw if binary else json.loads(raw)


def fetch(run_id, name, m, manifest_sha):
    run = gh(f"repos/{release.REPO}/actions/runs/{run_id}")
    commit = run["head_sha"]
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "head-commit")
    seals = source_seals(m)
    seals[f"{PUBLICATION_DIR}/manifest-8192-DRAFT.json"] = manifest_sha
    checked = {}
    for path, digest in seals.items():
        blob = gh(f"repos/{release.REPO}/contents/{path}?ref={commit}")
        require(
            blob["type"] == "file" and blob["path"] == path and blob["encoding"] == "base64",
            "upstream-blob",
        )
        data = base64.b64decode(blob["content"])
        require(len(data) == blob["size"] and sha(data) == digest, "upstream-source-sha")
        if not path.endswith("manifest-8192-DRAFT.json"):
            checked[path] = digest
    run["source_blobs"] = checked
    listing = gh(f"repos/{release.REPO}/actions/runs/{run_id}/artifacts?per_page=100")
    require(listing["total_count"] == 1, "exact-single-run-artifact")
    rows = listing["artifacts"]
    require(
        len(rows) == 1
        and type(rows[0]["id"]) is int
        and rows[0]["name"] == name
        and 0 < rows[0]["size_in_bytes"] <= 100000,
        "artifact-id-or-bound",
    )
    raw = gh(f"repos/{release.REPO}/actions/artifacts/{rows[0]['id']}/zip", True)
    return run, rows, raw


def temporary_inventory(m, reader=gh):
    """No artifact deletion; matching retry must reuse its successful original run."""
    prefix = f"fresh8192-{m['capsule_sha256']}-"
    total = 0
    names = set()
    for page in range(1, 11):
        listing = reader(f"repos/{release.REPO}/actions/artifacts?per_page=100&page={page}")
        rows = listing["artifacts"]
        require(isinstance(rows, list) and len(rows) <= 100, "artifact-pagination")
        for row in rows:
            if not row["expired"] and row["name"].startswith(prefix):
                require(
                    row["name"] not in names
                    and type(row["size_in_bytes"]) is int
                    and 0 < row["size_in_bytes"] <= 100000,
                    "duplicate-temp-artifact",
                )
                names.add(row["name"])
                total += row["size_in_bytes"]
        require(total <= TEMP_LIMIT, "existing-temp-quota")
        if len(rows) < 100:
            return names, total
    raise ValueError("artifact-inventory-incomplete")


def main():
    global ACTIVE_CONTROL
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
    control = json.loads((ROOT / "transport-control-PENDING.json").read_bytes())
    release.check_clock(control)
    ACTIVE_CONTROL = control
    if os.environ["MODE"] == "chunk":
        ordinal = int(os.environ["ORDINAL"])
        data = chunk_input(m, ordinal, os.environ["PAYLOAD"])
        names, used = temporary_inventory(m)
        require(artifact_name(m, ordinal) not in names, "reuse-existing-successful-run")
        require(used + 100000 <= TEMP_LIMIT, "projected-temp-quota")
        out = Path(os.environ["RUNNER_TEMP"]) / "fresh-capsule-chunk"
        out.mkdir(exist_ok=False)
        (out / "chunk.bin").write_bytes(data)
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"name={artifact_name(m, ordinal)}\npath={out}\n")
    else:
        require(os.environ["MODE"] == "aggregate", "mode")
        ids = json.loads(os.environ["RUN_IDS"])
        data = aggregate(m, ids, lambda ident, name: fetch(ident, name, m, digest), control)
        transport = release.Transport(os.environ["GH_TOKEN"], control)
        proof = release.publish_verified(transport, f"sha256-{sha(data)}.tar.gz", data)
        verify(transport.download(proof["name"], len(data)), m)
        print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
