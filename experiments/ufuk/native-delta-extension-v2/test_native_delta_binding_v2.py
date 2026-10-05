"""Exact raw-file fixtures; no Torch, network, inference or optimizer work."""

import json
import os
from pathlib import Path

import native_delta_binding_v2 as codec
import native_delta_receiver_v2 as receiver
import pytest

PARENT = Path(
    "/workspace/work/harbichess/a100/restoration/local-rehearsal-content/harbichess-inputs/initial-e8.safetensors"
)
PAYLOAD_ROOT = Path(os.environ.get("HARBICHESS_NATIVE_V2_FIXTURES", str(receiver.ROOT)))


@pytest.mark.parametrize("manifest_sha", list(receiver.ALLOWLIST))
def test_each_final_three_files_roundtrip_and_contract_preserved(manifest_sha):
    manifest = receiver.load_manifest(manifest_sha)
    folder = PAYLOAD_ROOT / receiver.ALLOWLIST[manifest_sha]
    chunks = [
        (folder.parent / f"chunk-{i:03d}.base64").read_text()
        for i in range(len(manifest["chunks"]))
    ]
    packet = codec.reassemble(manifest, chunks)
    files = codec.decode(packet, PARENT.read_bytes(), manifest["packet_sha256"], manifest["files"])
    assert codec.native_binding(files) == manifest["binding"]
    assert json.loads(files["checkpoint.json"])["accepted"] == 1024
    capsule = receiver.deterministic_capsule(files)
    receiver.verify_capsule(capsule, manifest["files"])
    bad = dict(files)
    contract = json.loads(files["checkpoint.json"])
    contract["contract"]["source_commit"] = "0" * 40
    bad["checkpoint.json"] = codec.canonical(contract)
    with pytest.raises(ValueError, match="fixed-family"):
        codec.native_binding(bad)


def test_v1_allowlist_not_accepted():
    with pytest.raises(receiver.Failure, match="fixed-allowlist"):
        receiver.load_manifest("170f4737816346e57d01f6d6fcceec48aa4e09fd8a364cccd7abcb21a4df4cf5")


def test_private_codec_does_not_mutate_original_v1_module():
    import native_delta_codec as original

    assert original.native_binding is not codec.native_binding
    assert original.native_binding.__module__ == "native_delta_codec"
    assert codec.digest((receiver.ROOT / "native_delta_codec.py").read_bytes()) == codec.CODEC_SHA
