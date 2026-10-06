"""Part integrity and bounds; no real native/model/network calls."""

import base64
import copy
import hashlib
import json

import parts_receiver_v1 as p
import pytest


def fixture(monkeypatch):
    monkeypatch.setattr(p.release, "check_clock", lambda _: None)
    c = dict(
        schema="NNUE-V9-exact-capsule-eight-parts-v1",
        status="ROOT-approved-before-part-POST",
        original_manifest_sha256=p.capsule.sha(b"{}"),
        part_limit_bytes=8 * 1024**2,
        max_parts=8,
        release_id=404068972,
        release_tag="ufuk-cpu-ownplay-state-20261005",
        full_capsule_sha256="a" * 64,
        full_capsule_bytes=8,
        clock={},
        parts=[],
    )
    for i in range(8):
        c["parts"].append(
            dict(
                index=i,
                offset=i,
                bytes=1,
                sha256="b" * 64,
                git_blob_sha1="c" * 40,
                asset_name=f"nnue-state-v9-aaaaaaaaaaaa-part{i:02d}-bbbbbbbbbbbb.bin",
            )
        )
    return c


def test_bounded_complete_order(monkeypatch):
    c = fixture(monkeypatch)
    p.validate(c, b"{}")
    c["parts"][3]["offset"] += 1
    with pytest.raises(ValueError):
        p.validate(c, b"{}")


@pytest.mark.parametrize("mutation", ["partsize", "total", "count", "manifest", "asset"])
def test_part_manifest_tampering(monkeypatch, mutation):
    c = copy.deepcopy(fixture(monkeypatch))
    if mutation == "partsize":
        c["parts"][0]["bytes"] = 8 * 1024**2 + 1
    elif mutation == "total":
        c["full_capsule_bytes"] += 1
    elif mutation == "count":
        c["parts"].pop()
    elif mutation == "manifest":
        c["original_manifest_sha256"] = "0" * 64
    else:
        c["parts"][0]["asset_name"] = c["parts"][1]["asset_name"]
    with pytest.raises(ValueError):
        p.validate(c, b"{}")


def test_git_part_exact_bytes_and_content_hashes():
    raw = b"known part bytes"
    row = dict(
        bytes=len(raw),
        sha256=p.capsule.sha(raw),
        git_blob_sha1=hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
    )
    packet = dict(
        sha=row["git_blob_sha1"],
        size=len(raw),
        encoding="base64",
        content=base64.b64encode(raw).decode(),
    )
    assert p.git_bytes(json.dumps(packet), row) == raw
    packet["content"] = base64.b64encode(b"x" * len(raw)).decode()
    with pytest.raises(ValueError):
        p.git_bytes(json.dumps(packet), row)
