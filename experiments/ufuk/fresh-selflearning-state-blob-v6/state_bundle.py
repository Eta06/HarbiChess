"""Exact 22-file fresh state bundle; standard-library lossless decoding only."""

import hashlib
import json
import struct
import zlib

import fresh_binding

SCHEMA = "fresh-selflearning-state-lossless-bundle-v5"
TAGS = tuple(f"{arm}-{seed}" for arm in ("mc", "sc", "fullcritic") for seed in (20262805, 20262806))
DATA_PATHS = tuple(
    f"data/{seed}/{name}"
    for seed in (20262805, 20262806)
    for name in ("actions-00008192.json.gz", "actor-config.json")
)
PATHS = tuple(f"{tag}/{name}" for tag in TAGS for name in fresh_binding.FILES) + DATA_PATHS
RAW_LIMIT = 8 * 1024**2
ENCODED_LIMIT = 4 * 1024**2
FILE_LIMIT = 2 * 1024**2
CHUNK = 43500
READINESS = "90162446656105007ca713c17b01f4e0a762d694a179a5be815e0ba6174bae51"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(value, code):
    if not value:
        raise ValueError(code)


def file_rows(originals):
    require(set(originals) == set(PATHS) and len(originals) == 22, "fixed-22-original-paths")
    rows = {p: {"bytes": len(originals[p]), "sha256": sha(originals[p])} for p in PATHS}
    require(
        all(0 < row["bytes"] <= FILE_LIMIT for row in rows.values()),
        "original-file2MiB",
    )
    require(sum(row["bytes"] for row in rows.values()) <= RAW_LIMIT, "original-total8MiB")
    return rows


def member_binding(tag, originals):
    result = fresh_binding.binding(originals)
    arm, seed = tag.rsplit("-", 1)
    expected = {
        "mc": "fresh-qsearch-additive-own-mc-native-v2",
        "sc": "fresh-qsearch-additive-own-sc-native-v1",
        "fullcritic": "fresh-qsearch-full-critic-own-search-native-v1",
    }
    require(
        result["schema"] == expected[arm] and result["seed"] == int(seed), "tag-native-schema-seed"
    )
    return result


def encode(originals, parent):
    rows = file_rows(originals)
    codec = fresh_binding.codec_instance()
    blocks, members = [], []
    for tag in TAGS:
        raw = {name: originals[f"{tag}/{name}"] for name in fresh_binding.FILES}
        packet = codec.encode(raw, parent)
        members.append(
            {
                "tag": tag,
                "bytes": len(packet),
                "sha256": sha(packet),
                "binding": member_binding(tag, raw),
            }
        )
        blocks.append(packet)
    for path in DATA_PATHS:
        blocks.append(originals[path])
    meta = {
        "schema": SCHEMA,
        "codec_sha256": fresh_binding.CODEC_SHA,
        "parent_sha256": codec.E8_SHA256,
        "readiness_sha256": READINESS,
        "native_members": members,
        "files": rows,
    }
    header = canonical(meta)
    require(len(header) <= 128 * 1024, "metadata128KiB")
    capsule = zlib.compress(struct.pack("<I", len(header)) + header + b"".join(blocks), 9)
    require(len(capsule) <= ENCODED_LIMIT, "encoded4MiB")
    manifest = {
        **meta,
        "capsule_bytes": len(capsule),
        "capsule_sha256": sha(capsule),
        "raw_bytes": sum(row["bytes"] for row in rows.values()),
        "chunks": [
            {
                "ordinal": i // CHUNK,
                "bytes": len(capsule[i : i + CHUNK]),
                "sha256": sha(capsule[i : i + CHUNK]),
            }
            for i in range(0, len(capsule), CHUNK)
        ],
    }
    return capsule, manifest


def validate(manifest):
    require(
        manifest["schema"] == SCHEMA
        and manifest["codec_sha256"] == fresh_binding.CODEC_SHA
        and manifest["readiness_sha256"] == READINESS
        and manifest["parent_sha256"] == fresh_binding.codec_instance().E8_SHA256,
        "schema-source-parent-readiness",
    )
    rows = manifest["files"]
    require(set(rows) == set(PATHS) and len(rows) == 22, "fixed-22-original-paths")
    require(
        all(
            type(row["bytes"]) is int
            and 0 < row["bytes"] <= FILE_LIMIT
            and valid_sha(row["sha256"])
            for row in rows.values()
        ),
        "file-size-sha",
    )
    require(
        sum(row["bytes"] for row in rows.values()) == manifest["raw_bytes"] <= RAW_LIMIT,
        "raw-sum8MiB",
    )
    size = manifest["capsule_bytes"]
    require(
        type(size) is int and 0 < size <= ENCODED_LIMIT and valid_sha(manifest["capsule_sha256"]),
        "capsule-size-sha",
    )
    chunks = manifest["chunks"]
    require(len(chunks) == (size + CHUNK - 1) // CHUNK, "chunk-count")
    require([r["ordinal"] for r in chunks] == list(range(len(chunks))), "chunk-ordinals")
    require(
        all(
            type(r["bytes"]) is int
            and r["bytes"] == min(CHUNK, size - i * CHUNK)
            and valid_sha(r["sha256"])
            for i, r in enumerate(chunks)
        ),
        "chunk-size-sha",
    )
    require([r["tag"] for r in manifest["native_members"]] == list(TAGS), "six-native-order")


def valid_sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def decode(capsule, parent, manifest):
    validate(manifest)
    require(
        len(capsule) == manifest["capsule_bytes"] and sha(capsule) == manifest["capsule_sha256"],
        "capsule-full-sha",
    )
    decompressor = zlib.decompressobj()
    data = decompressor.decompress(capsule, RAW_LIMIT + 128 * 1024 + 1)
    require(
        decompressor.eof
        and not decompressor.unused_data
        and not decompressor.unconsumed_tail
        and len(data) <= RAW_LIMIT + 128 * 1024,
        "bounded-single-zlib-frame",
    )
    require(len(data) >= 4, "frame-header")
    length = struct.unpack("<I", data[:4])[0]
    require(0 < length <= 128 * 1024 and length + 4 <= len(data), "metadata-size")
    meta = json.loads(data[4 : 4 + length])
    expected = {
        k: manifest[k]
        for k in (
            "schema",
            "codec_sha256",
            "parent_sha256",
            "readiness_sha256",
            "native_members",
            "files",
        )
    }
    require(meta == expected, "external-sealed-metadata")
    codec, cursor, restored = fresh_binding.codec_instance(), 4 + length, {}
    for member in meta["native_members"]:
        size = member["bytes"]
        require(
            type(size) is int and 0 < size <= ENCODED_LIMIT and cursor + size <= len(data),
            "native-packet-size",
        )
        packet = data[cursor : cursor + size]
        cursor += size
        rows = {name: meta["files"][f"{member['tag']}/{name}"] for name in fresh_binding.FILES}
        raw = codec.decode(packet, parent, member["sha256"], rows)
        require(
            member_binding(member["tag"], raw) == member["binding"],
            "exact-original-native-contract",
        )
        restored.update({f"{member['tag']}/{name}": value for name, value in raw.items()})
    for path in DATA_PATHS:
        size = meta["files"][path]["bytes"]
        require(cursor + size <= len(data), "data-size")
        restored[path] = data[cursor : cursor + size]
        cursor += size
    require(
        cursor == len(data) and file_rows(restored) == meta["files"],
        "all22-full-byte-shas",
    )
    # Contracts bind the dataset and actor config; prior fit clocks are unchanged literal bytes.
    for member in meta["native_members"]:
        c = member["binding"]["original_contract"]
        seed = c["seed"]
        require(
            c["journal_sha256"] == sha(restored[f"data/{seed}/actions-00008192.json.gz"])
            and c["actor_config_sha256"] == sha(restored[f"data/{seed}/actor-config.json"]),
            "native-external-dataset-config-bindings",
        )
    return restored
