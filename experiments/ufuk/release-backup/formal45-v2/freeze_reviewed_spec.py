"""Freeze only actual captured remote bytes explicitly approved by ROOT; no transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(pending, request, response, approval):
    assert request["schema"] == "formal45-evidence-resolution-request-v1"
    assert response["schema"] == "formal45-exact-file-review-v1"
    assert approval["schema"] == "formal45-public-evidence-byte-approval-v1"
    assert approval["status"] == "approved-exact-public-byte-allowlist"
    captured = {row["path"]: row for row in response["records"]}
    expected = {row["path"]: row for row in request["known"] + request["unresolved"]}
    approved = approval["reviewed_public_sha256"]
    assert set(expected) <= set(captured)
    rows = []
    for path, item in expected.items():
        record = captured[path]
        assert record["status"] == "exact-regular-file-read", path
        assert not item.get("expected_sha256") or record["sha256"] == item["expected_sha256"]
        assert approved[path] == record["sha256"], path
        if record.get("exact_local_copy"):
            assert sha(record["exact_local_copy"]) == record["sha256"]
        rows.append(
            {
                "path": path,
                "sha256": record["sha256"],
                "purpose": item["purpose"],
                "reviewed_public": True,
            }
        )
    # Additional failed partial/native/control files must be explicitly captured
    # and approved; directory recursion, reconstructed hashes and defaults forbidden.
    for path, purpose in approval.get("additional_evidence_purpose", {}).items():
        assert path not in expected
        record = captured[path]
        assert record["status"] == "exact-regular-file-read"
        assert approved[path] == record["sha256"]
        rows.append(
            {"path": path, "sha256": record["sha256"], "purpose": purpose, "reviewed_public": True}
        )
    barrier = approval["backup_barrier"]
    assert barrier["kind"] == "failed5-terminal-owned-release-v1"
    for role in ("terminal", "release"):
        path = barrier[role + "_path"]
        assert captured[path]["status"] == "exact-regular-file-read"
        assert barrier[role + "_sha256"] == captured[path]["sha256"] == approved[path]
        assert any(row["path"] == path for row in rows), "Barrier must itself be archived evidence"
    result = json.loads(json.dumps(pending))
    result["status"] = "frozen-reviewed-artifact-allowlist"
    result["backup_barrier"] = barrier
    for job in result["jobs"].values():
        job["reviewed_public_evidence"] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pending", "request", "response", "approval"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    objects = {}
    for name in ("pending", "request", "response", "approval"):
        path = getattr(args, name)
        assert sha(path) == getattr(args, name + "_sha256")
        objects[name] = json.loads(path.read_text())
    result = freeze(**objects)
    result["capture_response_sha256"] = args.response_sha256
    result["root_public_byte_approval_sha256"] = args.approval_sha256
    raw = (json.dumps(result, indent=2) + "\n").encode()
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": "frozen-exact-reviewed-spec-no-execution",
                "spec_sha256": hashlib.sha256(raw).hexdigest(),
                "public_evidence_count": len(
                    result["jobs"]["20261425"]["reviewed_public_evidence"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
