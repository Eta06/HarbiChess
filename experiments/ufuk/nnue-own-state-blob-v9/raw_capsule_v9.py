"""Explicit regular-file byte capsule; no native decoding, file writes or transport."""

import hashlib
import io
import json
import re
import struct
import zlib
from pathlib import Path

MAGIC = b"UFUKRAW9\0"
SCHEMA = "classical-explicit-raw-native-capsule-v9"
RAW_LIMIT = 384 * 1024**2
ENCODED_LIMIT = 64 * 1024**2
FILE_LIMIT = 8 * 1024**2


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_rows(manifest):
    if manifest.get("schema") != SCHEMA or manifest.get("approval") != "ROOT-approved-exact-files":
        raise ValueError("unapproved new explicit scope")
    if manifest.get("limits") != {
        "files": 2048,
        "raw_bytes": RAW_LIMIT,
        "encoded_bytes": ENCODED_LIMIT,
        "file_bytes": FILE_LIMIT,
        "header_bytes": 1024 * 1024,
    }:
        raise ValueError("V9 larger prospective bounds require explicit approval")
    rows = manifest["rows"]
    if manifest.get("row_count") != len(rows) or manifest.get("raw_bytes") != sum(
        r["bytes"] for r in rows
    ):
        raise ValueError("exact frozen measurement")
    if not rows or len(rows) > 2048:
        raise ValueError("empty/oversize allowlist")
    members = set()
    paths = set()
    for row in rows:
        if set(row) != {"path", "member", "role", "bytes", "sha256"}:
            raise ValueError("row fields")
        if not re.fullmatch(r"files/[0-9]{4}", row["member"]) or row["member"] in members:
            raise ValueError("member alias/traversal")
        if row["path"] in paths or not Path(row["path"]).is_absolute():
            raise ValueError("duplicate/nonabsolute original path")
        if not isinstance(row["bytes"], int) or not 0 <= row["bytes"] <= FILE_LIMIT:
            raise ValueError("file bound")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("sha format")
        members.add(row["member"])
        paths.add(row["path"])
    if sum(row["bytes"] for row in rows) > RAW_LIMIT:
        raise ValueError("raw384MiB: larger scope needs new prospective approval/code")
    return rows


def encode(manifest, guard=None):
    rows = checked_rows(manifest)
    compressor = zlib.compressobj(9)
    output = bytearray()
    header = json.dumps(
        {"schema": SCHEMA, "rows": rows}, sort_keys=True, separators=(",", ":")
    ).encode()
    if len(header) > 1024 * 1024:
        raise ValueError("header1MiB")
    output.extend(compressor.compress(MAGIC + struct.pack(">I", len(header)) + header))
    for row in rows:
        p = Path(row["path"])
        if p.is_symlink() or not p.is_file():
            raise ValueError("nonregular input")
        digest = hashlib.sha256()
        size = 0
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                if guard is not None:
                    guard()
                size += len(block)
                digest.update(block)
                output.extend(compressor.compress(block))
                if size > row["bytes"] or len(output) > ENCODED_LIMIT:
                    raise ValueError("changing input/encoded bound")
        if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError("original bytes changed")
    output.extend(compressor.flush())
    if len(output) > ENCODED_LIMIT:
        raise ValueError("encoded bound")
    return bytes(output)


def verify(capsule, manifest):
    rows = checked_rows(manifest)
    if len(capsule) > ENCODED_LIMIT:
        raise ValueError("encoded bound")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(capsule, RAW_LIMIT + 1024 * 1024 + len(MAGIC) + 4 + 1)
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("truncated/trailing/decompression bound")
    stream = io.BytesIO(raw)
    if stream.read(len(MAGIC)) != MAGIC:
        raise ValueError("magic")
    length = stream.read(4)
    if len(length) != 4:
        raise ValueError("header length")
    n = struct.unpack(">I", length)[0]
    if n > 1024 * 1024:
        raise ValueError("header bound")
    header = json.loads(stream.read(n))
    if header != {"schema": SCHEMA, "rows": rows}:
        raise ValueError("exact allowlist binding")
    result = []
    for row in rows:
        value = stream.read(row["bytes"])
        if len(value) != row["bytes"] or sha(value) != row["sha256"]:
            raise ValueError("original full SHA/size")
        result.append({"member": row["member"], "bytes": len(value), "sha256": sha(value)})
    if stream.read(1):
        raise ValueError("trailing raw bytes")
    return result


def digest(raw):
    return sha(raw)
