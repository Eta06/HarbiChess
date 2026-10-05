"""Lossless raw-file codec for sealed CPU VALUE/POSITION native backup.

No Torch import or reserialization: copies exact tensor/storage bytes by SHA from
public e8 or a preceding reconstructed model; preserves every other byte literally.
Networking, workflow dispatch and uploading are intentionally absent.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import struct
import zipfile
import zlib
from pathlib import Path

E8_SHA256 = "e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03"
PUBLIC_PARENT_ARCHIVE_SHA256 = "846446f3e64d7fde206c595dc50dcb8fce96fc6b2501af31fbb5186b599d64ee"
PUBLIC_PARENT_ARCHIVE_BYTES = 127167165
PUBLIC_PARENT_MEMBER = "payload/000128"
SOURCE_COMMIT = "4cae08522ac940746cf9127ca8ce4a6b63a47408"
FILES = ("model.safetensors", "training.pt", "checkpoint.json")
CHUNK_CHARS = 58000
MAX_RAW_BYTES = 16 * 1024**2
MAX_PACKET_BYTES = 4 * 1024**2
MAX_METADATA_BYTES = 1024**2
SCHEMA = "cpu-exact-native-delta-v1"
CHUNK_SCHEMA = "cpu-exact-native-delta-chunks-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checked_digest(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("invalid-sha256")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError("invalid-sha256")
    return value


def checked_int(value, maximum=MAX_RAW_BYTES):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("invalid-bounded-integer")
    return value


def tensor_spans(raw):
    """Validate safetensors geometry and return exact contiguous data spans."""
    if len(raw) < 8 or len(raw) > MAX_RAW_BYTES:
        raise ValueError("safetensors-size")
    size = struct.unpack("<Q", raw[:8])[0]
    if not 2 <= size <= MAX_METADATA_BYTES or 8 + size > len(raw):
        raise ValueError("safetensors-header-size")
    header = json.loads(raw[8 : 8 + size])
    spans = []
    widths = {
        "F32": 4,
        "F64": 8,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
        "U16": 2,
        "U32": 4,
        "U64": 8,
    }
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or item["dtype"] not in widths:
            raise ValueError("safetensors-tensor-description")
        start, end = item["data_offsets"]
        start, end = checked_int(start), checked_int(end)
        shape = item["shape"]
        if any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("safetensors-shape")
        if end < start or math.prod(shape) * widths[item["dtype"]] != end - start:
            raise ValueError("safetensors-shape-size")
        if 8 + size + end > len(raw):
            raise ValueError("safetensors-data-boundary")
        data = raw[8 + size + start : 8 + size + end]
        spans.append((8 + size + start, len(data), digest(data)))
    cursor = 8 + size
    for start, length, _ in sorted(spans):
        if start != cursor:
            raise ValueError("safetensors-data-not-contiguous")
        cursor += length
    if cursor != len(raw):
        raise ValueError("safetensors-trailing-data")
    return sorted(spans)


def tensor_dictionary(raw):
    return {(length, sha): raw[start : start + length] for start, length, sha in tensor_spans(raw)}


def checked_parent(raw):
    if digest(raw) != E8_SHA256:
        raise ValueError("public-parent-e8-sha-mismatch")
    return tensor_dictionary(raw)


def native_binding(raw_files):
    if tuple(raw_files) != FILES:
        raise ValueError("exact-three-native-files-in-order-required")
    if sum(len(raw_files[name]) for name in FILES) > MAX_RAW_BYTES:
        raise ValueError("native-file-size-bound")
    checkpoint = json.loads(raw_files["checkpoint.json"])
    allowed = {
        "cpu-own-outcome-linear-training-native-v2",
        "cpu-own-outcome-positional-training-native-v2",
    }
    contract = checkpoint["contract"]
    if checkpoint["schema"] not in allowed or contract["source_commit"] != SOURCE_COMMIT:
        raise ValueError("sealed-native-schema-source")
    if contract["initial_e8_sha256"] != E8_SHA256 or contract["device"] != "cpu":
        raise ValueError("sealed-native-initial-cpu-binding")
    if set(checkpoint["artifacts"]) != set(FILES[:2]):
        raise ValueError("sealed-native-artifact-set")
    for name in FILES[:2]:
        if digest(raw_files[name]) != checkpoint["artifacts"][name]:
            raise ValueError("sealed-native-artifact-sha")
    return {
        "schema": checkpoint["schema"],
        "source_commit": SOURCE_COMMIT,
        "initial_e8_sha256": E8_SHA256,
        "seed": contract["seed"],
        "accepted": checkpoint["accepted"],
        "torch_version": contract["torch_version"],
        "protocol_sha256": contract["protocol_sha256"],
        "helper_sha256": contract["helper_sha256"],
        "feature_helper_sha256": contract["feature_helper_sha256"],
        "journal_sha256": contract["journal_sha256"],
    }


def storage_spans(raw, dictionaries):
    spans = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = set()
        for info in archive.infolist():
            if info.filename in names:
                raise ValueError("duplicate-torch-zip-member")
            names.add(info.filename)
            if "/data/" not in info.filename or info.compress_type != zipfile.ZIP_STORED:
                continue
            if info.file_size > MAX_RAW_BYTES:
                raise ValueError("torch-storage-size-bound")
            data = archive.read(info)  # Checks ZIP CRC without unpickling anything.
            key = (len(data), digest(data))
            where = next((name for name, words in dictionaries.items() if key in words), None)
            if where is None:
                continue
            offset = info.header_offset
            if raw[offset : offset + 4] != b"PK\x03\x04":
                raise ValueError("torch-local-zip-header")
            name_size, extra_size = struct.unpack("<HH", raw[offset + 26 : offset + 30])
            start = offset + 30 + name_size + extra_size
            if raw[start : start + len(data)] != data:
                raise ValueError("torch-storage-raw-byte-mismatch")
            spans.append((start, len(data), where, key[1]))
    return sorted(spans)


def encode(raw_files, parent_raw):
    binding = native_binding(raw_files)
    dictionaries = {"parent": checked_parent(parent_raw)}
    literals, entries = bytearray(), []
    for name in FILES:
        raw = raw_files[name]
        if name == FILES[0]:
            spans = [
                (start, size, "parent", sha)
                for start, size, sha in tensor_spans(raw)
                if (size, sha) in dictionaries["parent"]
            ]
        elif name == FILES[1]:
            spans = storage_spans(raw, dictionaries)
        else:
            spans = []
        cursor, operations = 0, []
        for start, size, where, sha in spans:
            if start < cursor:
                raise ValueError("copy-spans-overlap")
            if start > cursor:
                operations.append(["literal", len(literals), start - cursor])
                literals.extend(raw[cursor:start])
            operations.append([where, sha, size])
            cursor = start + size
        if cursor < len(raw):
            operations.append(["literal", len(literals), len(raw) - cursor])
            literals.extend(raw[cursor:])
        entries.append(
            {"name": name, "bytes": len(raw), "sha256": digest(raw), "operations": operations}
        )
        if name == FILES[0]:
            dictionaries["model"] = tensor_dictionary(raw)
    metadata = {
        "schema": SCHEMA,
        "parent_sha256": E8_SHA256,
        "binding": binding,
        "files": entries,
        "literal_bytes": len(literals),
    }
    encoded = canonical(metadata)
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError("packet-metadata-bound")
    packet = zlib.compress(struct.pack("<I", len(encoded)) + encoded + literals, 9)
    if len(packet) > MAX_PACKET_BYTES:
        raise ValueError("packet-size-bound")
    return packet


def decode(packet, parent_raw, expected_packet_sha256, expected_files):
    checked_digest(expected_packet_sha256)
    if digest(packet) != expected_packet_sha256 or len(packet) > MAX_PACKET_BYTES:
        raise ValueError("packet-sha-or-size")
    dictionaries = {"parent": checked_parent(parent_raw)}
    inflater = zlib.decompressobj()
    payload = inflater.decompress(packet, MAX_RAW_BYTES + MAX_METADATA_BYTES + 5)
    if (
        not inflater.eof
        or inflater.unused_data
        or inflater.unconsumed_tail
        or len(payload) > MAX_RAW_BYTES + MAX_METADATA_BYTES + 4
    ):
        raise ValueError("bounded-single-zlib-stream-required")
    if len(payload) < 4:
        raise ValueError("packet-header")
    length = checked_int(struct.unpack("<I", payload[:4])[0], MAX_METADATA_BYTES)
    if length + 4 > len(payload):
        raise ValueError("packet-metadata-truncated")
    metadata = json.loads(payload[4 : 4 + length])
    literals = payload[4 + length :]
    if (
        metadata["schema"] != SCHEMA
        or metadata["parent_sha256"] != E8_SHA256
        or metadata["literal_bytes"] != len(literals)
    ):
        raise ValueError("packet-schema-parent-literal-binding")
    entries = metadata["files"]
    if tuple(row["name"] for row in entries) != FILES or set(expected_files) != set(FILES):
        raise ValueError("exact-native-filenames-required")
    if sum(checked_int(row["bytes"]) for row in entries) > MAX_RAW_BYTES:
        raise ValueError("combined-native-size-bound")
    restored, literal_cursor = {}, 0
    for entry in entries:
        expected = expected_files[entry["name"]]
        target_size = checked_int(entry["bytes"])
        if target_size != expected["bytes"] or entry["sha256"] != expected["sha256"]:
            raise ValueError("external-sealed-native-file-binding")
        output = bytearray()
        for kind, key, size in entry["operations"]:
            size = checked_int(size)
            if len(output) + size > target_size:
                raise ValueError("operation-target-size-bound")
            if kind == "literal":
                start = checked_int(key)
                if start != literal_cursor or start + size > len(literals):
                    raise ValueError("literal-contiguous-boundary")
                data = literals[start : start + size]
                literal_cursor += size
            elif kind in dictionaries:
                checked_digest(key)
                data = dictionaries[kind].get((size, key))
                if data is None or digest(data) != key:
                    raise ValueError("exact-tensor-content-sha-required")
            else:
                raise ValueError("unavailable-or-forward-copy-dictionary")
            output.extend(data)
        output = bytes(output)
        if len(output) != target_size or digest(output) != entry["sha256"]:
            raise ValueError("restored-original-file-byte-sha")
        restored[entry["name"]] = output
        if entry["name"] == FILES[0]:
            dictionaries["model"] = tensor_dictionary(output)
    if literal_cursor != len(literals):
        raise ValueError("unconsumed-literal-data")
    if native_binding(restored) != metadata["binding"]:
        raise ValueError("restored-native-contract-binding")
    return restored


def chunk_manifest(packet, raw_files, codec_sha256):
    text = base64.b64encode(packet).decode("ascii")
    chunks = [text[start : start + CHUNK_CHARS] for start in range(0, len(text), CHUNK_CHARS)]
    manifest = {
        "schema": CHUNK_SCHEMA,
        "packet_sha256": digest(packet),
        "packet_bytes": len(packet),
        "parent_sha256": E8_SHA256,
        "parent_archive_sha256": PUBLIC_PARENT_ARCHIVE_SHA256,
        "parent_archive_bytes": PUBLIC_PARENT_ARCHIVE_BYTES,
        "parent_archive_member": PUBLIC_PARENT_MEMBER,
        "codec_sha256": checked_digest(codec_sha256),
        "chunk_chars_max": CHUNK_CHARS,
        "files": {
            name: {"bytes": len(raw_files[name]), "sha256": digest(raw_files[name])}
            for name in FILES
        },
        "binding": native_binding(raw_files),
        "chunks": [
            {"ordinal": index, "characters": len(chunk), "sha256": digest(chunk.encode("ascii"))}
            for index, chunk in enumerate(chunks)
        ],
    }
    return manifest, chunks


def reassemble(manifest, chunks):
    if (
        manifest["schema"] != CHUNK_SCHEMA
        or manifest["parent_sha256"] != E8_SHA256
        or manifest["chunk_chars_max"] != CHUNK_CHARS
        or manifest["parent_archive_sha256"] != PUBLIC_PARENT_ARCHIVE_SHA256
        or manifest["parent_archive_bytes"] != PUBLIC_PARENT_ARCHIVE_BYTES
        or manifest["parent_archive_member"] != PUBLIC_PARENT_MEMBER
    ):
        raise ValueError("sealed-chunk-manifest-schema-parent")
    checked_digest(manifest["codec_sha256"])
    if not 1 <= len(chunks) == len(manifest["chunks"]) <= 100:
        raise ValueError("exact-chunk-count")
    for index, (chunk, entry) in enumerate(zip(chunks, manifest["chunks"], strict=True)):
        if (
            entry["ordinal"] != index
            or not isinstance(chunk, str)
            or len(chunk) != entry["characters"]
            or not 0 < len(chunk) <= CHUNK_CHARS
            or digest(chunk.encode("ascii")) != entry["sha256"]
        ):
            raise ValueError("exact-chunk-ordinal-length-sha")
        if index < len(chunks) - 1 and len(chunk) != CHUNK_CHARS:
            raise ValueError("canonical-nonfinal-chunk-length")
    packet = base64.b64decode("".join(chunks), validate=True)
    if len(packet) != manifest["packet_bytes"] or digest(packet) != manifest["packet_sha256"]:
        raise ValueError("assembled-packet-size-sha")
    return packet


def dispatch_envelope(manifest, chunks, ordinal, frozen_manifest_sha256, ref="main"):
    """Build a bounded JSON request only; never sends or dispatches anything."""
    if digest(canonical(manifest) + b"\n") != frozen_manifest_sha256:
        raise ValueError("frozen-chunk-manifest-sha")
    reassemble(manifest, chunks)
    if type(ordinal) is not int or not 0 <= ordinal < len(chunks):
        raise ValueError("dispatch-ordinal")
    inputs = {
        "manifest_sha256": frozen_manifest_sha256,
        "packet_sha256": manifest["packet_sha256"],
        "ordinal": str(ordinal),
        "count": str(len(chunks)),
        "chunk_sha256": manifest["chunks"][ordinal]["sha256"],
        "payload_base64": chunks[ordinal],
    }
    if len(canonical(inputs)) > 65535 or ref != "main":
        raise ValueError("fixed-main-bounded-dispatch-inputs")
    return {"ref": ref, "inputs": inputs}


def regular_read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RAW_BYTES:
        raise ValueError("regular-bounded-file-required")
    return path.read_bytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    originals = {name: regular_read(args.native / name) for name in FILES}
    parent = regular_read(args.parent)
    packet = encode(originals, parent)
    manifest, chunks = chunk_manifest(packet, originals, digest(Path(__file__).read_bytes()))
    decoded = decode(
        reassemble(manifest, chunks), parent, manifest["packet_sha256"], manifest["files"]
    )
    if decoded != originals:
        raise ValueError("roundtrip-original-bytes-not-identical")
    total = len(packet) + len(canonical(manifest)) + sum(len(chunk) for chunk in chunks)
    if total > 2 * 1024**2:
        raise ValueError("bounded-proposal-output-size")
    args.output.mkdir(exist_ok=False)
    (args.output / "packet.zlib").write_bytes(packet)
    (args.output / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    for index, chunk in enumerate(chunks):
        (args.output / f"chunk-{index:03d}.base64").write_text(chunk, encoding="ascii")
    print(
        json.dumps(
            {
                "status": "local-exact-byte-roundtrip-pass-public-pending",
                "packet_bytes": len(packet),
                "chunks": len(chunks),
                "packet_sha256": manifest["packet_sha256"],
                "native_schema": manifest["binding"]["schema"],
                "native_source_commit": SOURCE_COMMIT,
                "no_torch_or_network_or_training": True,
            }
        )
    )


if __name__ == "__main__":
    main()
