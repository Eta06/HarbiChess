"""Exact finalized native fixtures; no model execution or fixture mutation."""

import copy
import hashlib
import json
import struct
import zlib
from pathlib import Path

import native_delta_binding_v2 as codec
import pytest

PARENT = Path(
    "/workspace/work/harbichess/a100/restoration/local-rehearsal-content/"
    "harbichess-inputs/initial-e8.safetensors"
)
FINALS = [
    Path("/workspace/work/harbichess/cpu-shrunk-value-v1-actual/fits")
    / f"{seed}-shrunk/checkpoints/step-00001024"
    for seed in (20262605, 20262606)
] + [
    Path("/workspace/work/harbichess/cpu-residual-value-v1-actual/fits")
    / f"{seed}-residual/checkpoints/step-00001024"
    for seed in (20262705, 20262706)
]


@pytest.fixture(scope="module")
def value_fixture():
    originals = {name: (FINALS[0] / name).read_bytes() for name in codec.FILES}
    parent = PARENT.read_bytes()
    packet = codec.encode(originals, parent)
    manifest, chunks = codec.chunk_manifest(packet, originals, "a" * 64)
    return originals, parent, packet, manifest, chunks


@pytest.mark.parametrize("native", FINALS)
def test_all_three_actual_final_files_have_original_storage_bytes_after_roundtrip(native):
    originals = {name: (native / name).read_bytes() for name in codec.FILES}
    parent = PARENT.read_bytes()
    packet = codec.encode(originals, parent)
    manifest, chunks = codec.chunk_manifest(packet, originals, "a" * 64)
    assert all(0 < len(chunk) <= 58000 for chunk in chunks)
    assert codec.reassemble(manifest, chunks) == packet
    restored = codec.decode(packet, parent, manifest["packet_sha256"], manifest["files"])
    assert restored == originals
    assert set(restored) == {"model.safetensors", "training.pt", "checkpoint.json"}
    assert all(
        hashlib.sha256(restored[name]).hexdigest() == manifest["files"][name]["sha256"]
        for name in codec.FILES
    )
    # A smaller packet proves the real frozen storage path was used, beyond a raw ZIP wrapper.
    assert len(packet) < sum(len(value) for value in originals.values()) / 2


@pytest.mark.parametrize("mutation", ["missing", "swapped", "byte", "ordinal"])
def test_chunk_inventory_fails_closed_on_missing_reordered_or_corrupted_data(
    value_fixture, mutation
):
    _, _, _, manifest, chunks = value_fixture
    manifest, chunks = copy.deepcopy(manifest), list(chunks)
    if mutation == "missing":
        chunks.pop()
    elif mutation == "swapped":
        chunks.reverse()
    elif mutation == "byte":
        chunks[0] = ("B" if chunks[0][0] != "B" else "A") + chunks[0][1:]
    else:
        manifest["chunks"][0]["ordinal"] = 1
    with pytest.raises(ValueError):
        codec.reassemble(manifest, chunks)


def test_wrong_public_parent_and_external_original_file_binding_fail(value_fixture):
    _, parent, packet, manifest, _ = value_fixture
    with pytest.raises(ValueError, match="parent-e8-sha"):
        codec.decode(
            packet,
            parent[:-1] + bytes([parent[-1] ^ 1]),
            manifest["packet_sha256"],
            manifest["files"],
        )
    files = copy.deepcopy(manifest["files"])
    files["training.pt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="external-sealed-native-file-binding"):
        codec.decode(packet, parent, manifest["packet_sha256"], files)


def test_metadata_forward_copy_and_trailing_compressed_stream_fail(value_fixture):
    _, parent, packet, manifest, _ = value_fixture
    with pytest.raises(ValueError, match="single-zlib-stream"):
        codec.decode(
            packet + zlib.compress(b"extra"),
            parent,
            codec.digest(packet + zlib.compress(b"extra")),
            manifest["files"],
        )
    payload = zlib.decompress(packet)
    size = struct.unpack("<I", payload[:4])[0]
    metadata = json.loads(payload[4 : 4 + size])
    op = next(op for op in metadata["files"][0]["operations"] if op[0] == "parent")
    op[0] = "model"  # The model does not exist yet: no reference cycles permitted.
    encoded = codec.canonical(metadata)
    broken = zlib.compress(struct.pack("<I", len(encoded)) + encoded + payload[4 + size :])
    with pytest.raises(ValueError, match="forward-copy"):
        codec.decode(broken, parent, codec.digest(broken), manifest["files"])


def test_signed_zero_tensor_words_remain_distinct_sha_and_do_not_compare_as_floats():
    header = codec.canonical(
        {
            "positive": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "negative": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        }
    )
    raw = struct.pack("<Q", len(header)) + header + struct.pack("<ff", 0.0, -0.0)
    words = codec.tensor_dictionary(raw)
    assert len(words) == 2
    assert set(words.values()) == {b"\x00\x00\x00\x00", b"\x00\x00\x00\x80"}


def test_encoder_rejects_checkpoint_model_sha_mismatch(value_fixture):
    originals, parent, _, _, _ = value_fixture
    corrupted = dict(originals)
    raw = corrupted["model.safetensors"]
    corrupted["model.safetensors"] = raw[:-1] + bytes([raw[-1] ^ 1])
    with pytest.raises(ValueError, match="native-artifact-sha"):
        codec.encode(corrupted, parent)


def test_dispatch_envelope_fits_real_json_bound_and_is_frozen_manifest_bound(value_fixture):
    _, _, _, manifest, chunks = value_fixture
    sha = codec.digest(codec.canonical(manifest) + b"\n")
    for index in range(len(chunks)):
        body = codec.dispatch_envelope(manifest, chunks, index, sha)
        assert len(codec.canonical(body)) < 65535
        assert len(body["inputs"]["payload_base64"]) <= 58000
        assert body["ref"] == "main"
    with pytest.raises(ValueError, match="frozen-chunk-manifest-sha"):
        codec.dispatch_envelope(manifest, chunks, 0, "0" * 64)
    with pytest.raises(ValueError, match="fixed-main"):
        codec.dispatch_envelope(manifest, chunks, 0, sha, ref="different-ref")
