"""Construct reviewable dispatch inputs; never dispatch or read credentials."""

import json

import artifact_capsule as c


def dispatch_inputs(raw_manifest, approved_sha, mode, *, ordinal=None, payload="", run_ids=()):
    c.require(c.sha(raw_manifest) == approved_sha, "launcher-approved-manifest-sha")
    m = json.loads(raw_manifest)
    c.validate(m)
    inputs = dict(
        mode=mode,
        manifest_sha=approved_sha,
        manifest_prefix=approved_sha[:12],
        ordinal="0",
        payload="",
    )
    if mode == "chunk":
        c.chunk_input(m, ordinal, payload)
        inputs.update(ordinal=str(ordinal), payload=payload)
    else:
        c.require(mode == "aggregate" and payload == "", "launcher-mode")
        c.require(not run_ids, "release-route-no-artifact-run-ids")
    encoded = json.dumps({"ref": "main", "inputs": inputs}, separators=(",", ":")).encode()
    c.require(len(encoded) <= 65535, "total-dispatch-json-bound")
    return encoded
