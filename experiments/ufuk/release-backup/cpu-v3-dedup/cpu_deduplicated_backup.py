"""Sealed CPU-v3 regular-blob deduplication and selected-native streamed restore."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

SOURCE = "3be5b87db27a0fbde83464e7ea7157f0d9a76ae4"
END = 1791180000
MAX_TOTAL = 8 * 1024**3
MAX_ASSETS = 200
BLOCK = 1024**2
FILES = {
    "model.safetensors",
    "base.safetensors",
    "behavior.safetensors",
    "training.pt",
    "actor.json",
    "last-frozen-epoch.json.gz",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(BLOCK):
            h.update(b)
    return h.hexdigest()


def check_file(path, digest):
    if path.is_symlink() or not path.is_file() or sha(path) != digest:
        raise ValueError("sealed-file-changed-or-not-regular")


def guard(deadline):
    if time.time() >= min(deadline, END):
        raise RuntimeError("original-backup-deadline-expired")


class ReservedDiskReader:
    def __init__(self, stream, output, deadline):
        self.stream, self.output, self.deadline = stream, output, deadline

    def read(self, count):
        guard(self.deadline)
        if shutil.disk_usage(self.output).free < 128 * 1024**2:
            raise ValueError("reserved-disk-headroom-no-old-files-deleted")
        return self.stream.read(count)


def check_native(directory, rows):
    path = directory / "checkpoint.json"
    if str(path) not in rows:
        raise ValueError("unapproved-native-manifest")
    check_file(path, rows[str(path)]["sha256"])
    m = json.loads(path.read_text())
    if m["source_commit"] != SOURCE or m["schema"] != "torch-search-acting-native-cpu-v3":
        raise ValueError("source8-native-source-schema")
    if m["runtime"]["device"] != "cpu" or m["run_config"]["config"]["device"] != "cpu":
        raise ValueError("honest-cpu-runtime-config-required")
    state = m["state"]
    if (
        state["replay_buffer"] != "closed-empty"
        or state["training_pass"] != "closed"
        or state["pending_search_schedule"] != "closed-empty"
    ):
        raise ValueError("unclosed-native")
    if set(m["artifacts"]) != FILES:
        raise ValueError("six-native-payloads-required")
    for name, digest in m["artifacts"].items():
        item = rows.get(str(directory / name))
        if item is None or item["sha256"] != digest:
            raise ValueError("native-payload-missing-or-unsealed")
    if set(m["inputs"]) != {"book", "initial_weights", "experiment_config", "protocol"}:
        raise ValueError("four-original-inputs-required")
    for key, identity in m["inputs"].items():
        resolved = os.path.normpath(str(directory / identity["relative_path"]))
        if (
            resolved not in rows
            or rows[resolved]["sha256"] != identity["sha256"]
            or m["run_config"]["input_sha256"][key] != identity["sha256"]
        ):
            raise ValueError("input-companion-missing-or-unsealed")
    if state != json.loads((directory / "actor.json").read_text()):
        raise ValueError("native-actor-manifest-differs")
    epoch = state["epoch"]
    if directory.name != f"epoch-{epoch:08d}":
        raise ValueError("native-epoch-path-mismatch")
    if epoch:
        journal = directory.parent.parent / "journal" / f"epoch-{epoch:08d}.json.gz"
        if (
            str(journal) not in rows
            or rows[str(journal)]["sha256"] != m["artifacts"]["last-frozen-epoch.json.gz"]
        ):
            raise ValueError("native-frozen-journal-companion-mismatch")
        record = json.loads(gzip.decompress(journal.read_bytes()))
        if (
            record["sample_chain_sha256"] != state["sample_chain_sha256"]
            or record["schema"] != m["run_config"]["schema"]
            or record["epoch"] != epoch
        ):
            raise ValueError("journal-native-chain-schema-epoch-mismatch")
    return m


def validate_spec(spec):
    if (
        spec.get("schema") != "cpu-v3-sealed-prefix-public-backup-spec-v1"
        or spec.get("status") != "approved-exact-public-byte-allowlist"
        or spec.get("source_commit") != SOURCE
    ):
        raise ValueError("explicit-source8-approved-spec-required")
    rows = {r["path"]: r for r in spec["files"]}
    if len(rows) != len(spec["files"]):
        raise ValueError("duplicate-approved-file")
    for path, row in rows.items():
        if (
            not any(Path(path).is_relative_to(Path(root)) for root in spec["approved_roots"])
            or ".." in Path(path).parts
            or "\\" in path
            or row["reviewed_public"] is not True
        ):
            raise ValueError("unreviewed-or-unsafe-original-path")
    groups = spec["groups"]
    if not groups or len(groups) > 100:
        raise ValueError("bounded-archive-groups-required")
    if not any(g.get("include_source_bundle") is True for g in groups):
        raise ValueError("original-source-bundle-must-be-publicly-archived")
    selected = set()
    for group in groups:
        if not group["files"] or len(group["files"]) != len(set(group["files"])):
            raise ValueError("nonempty-unique-group-files-required")
        for path in group["files"]:
            if path not in rows:
                raise ValueError("group-file-not-approved")
            selected.add(path)
    if selected != set(rows):
        raise ValueError("approved-file-unarchived")
    return rows


def pack(spec_path, spec_sha, prior_path, output, deadline):
    guard(deadline)
    check_file(spec_path, spec_sha)
    spec = json.loads(spec_path.read_text())
    rows = validate_spec(spec)
    check_file(prior_path, spec["prior_transport_manifest_sha256"])
    prior = json.loads(prior_path.read_text())
    if len(prior["assets"]) != 128:
        raise ValueError("prior128-required-no-overwrite")
    for path, row in rows.items():
        guard(deadline)
        check_file(Path(path), row["sha256"])
    native_manifests = [check_native(Path(p), rows) for p in spec["native_directories"]]
    prefixes = {}
    for directory, m in zip(spec["native_directories"], native_manifests, strict=True):
        prefixes.setdefault(str(Path(directory).parent), []).append(m["state"]["epoch"])
    if any(sorted(epochs) != list(range(max(epochs) + 1)) for epochs in prefixes.values()):
        raise ValueError("sealed-native-prefix-not-contiguous-from-zero")
    repo = Path(spec["source_repo"])
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != SOURCE
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    ):
        raise ValueError("original-source8-clean-producer-required")
    if shutil.disk_usage(output.parent).free < 256 * 1024**2:
        raise ValueError("insufficient-disk-headroom-no-old-files-deleted")
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "source.bundle"
    subprocess.run(
        ["git", "-c", "pack.threads=1", "bundle", "create", str(bundle), "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "bundle", "verify", str(bundle)],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    archives = []
    for number, group in enumerate(spec["groups"]):
        guard(deadline)
        if shutil.disk_usage(output).free < 128 * 1024**2:
            raise ValueError("reserved-disk-headroom-no-old-files-deleted")
        entries, blob_paths = make_blob_entries(group["files"], rows)
        source_digest = sha(bundle) if group.get("include_source_bundle") else None
        if source_digest:
            blob_paths.setdefault("blobs/" + source_digest, bundle)
        ledger = {
            "schema": "cpu-v3-content-addressed-regular-blob-map-v1",
            "source_commit": SOURCE,
            "spec_sha256": spec_sha,
            "group": group["name"],
            "files": entries,
            "source_bundle_sha256": source_digest,
            "source_bundle_bytes": bundle.stat().st_size if source_digest else None,
            "native_manifests": {
                directory: manifest
                for directory, manifest in zip(
                    spec["native_directories"], native_manifests, strict=True
                )
                if str(Path(directory) / "checkpoint.json") in group["files"]
            },
        }
        data = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode()
        pending = output / f"pending-{number:03d}.tar.gz"
        with pending.open("xb") as raw:
            with (
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as gz,
                tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as archive,
            ):
                info = tarfile.TarInfo("artifact-map.json")
                info.size = len(data)
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(data))
                members = [
                    (name, path, name.removeprefix("blobs/")) for name, path in blob_paths.items()
                ]
                for name, path, _ in members:
                    guard(deadline)
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mode = 0o600
                    with path.open("rb") as f:
                        archive.addfile(info, ReservedDiskReader(f, output, deadline))
            raw.flush()
            os.fsync(raw.fileno())
        if pending.stat().st_size > 1024**3:
            raise ValueError("archive-exceeds-unchanged-1GiB-cap")
        expected = {name: digest for name, _, digest in members}
        expected["artifact-map.json"] = hashlib.sha256(data).hexdigest()
        seen = {}
        with tarfile.open(pending, "r|gz") as archive:
            for member in archive:
                guard(deadline)
                if not member.isreg() or member.name in seen:
                    raise ValueError("unexpected-link-or-duplicate-member")
                h = hashlib.sha256()
                with archive.extractfile(member) as f:
                    while b := f.read(BLOCK):
                        h.update(b)
                seen[member.name] = h.hexdigest()
        if seen != expected:
            raise ValueError("complete-member-sha-proof-mismatch")
        digest = sha(pending)
        final = output / f"sha256-{digest}.tar.gz"
        pending.rename(final)
        final.chmod(0o600)
        archives.append(
            {
                "name": final.name,
                "bytes": final.stat().st_size,
                "sha256": digest,
                "group": group["name"],
            }
        )
    if (
        len(prior["assets"]) + len(archives) > MAX_ASSETS
        or sum(r["bytes"] for r in prior["assets"]) + sum(r["bytes"] for r in archives) > MAX_TOTAL
    ):
        raise ValueError("unchanged-aggregate-public-caps-exceeded")
    for path, row in rows.items():
        guard(deadline)
        check_file(Path(path), row["sha256"])
    result = {
        "schema": "cpu-v3-deduplicated-prefix-pack-result-v1",
        "status": "pack-pass-public-pending",
        "source_commit": SOURCE,
        "spec_sha256": spec_sha,
        "sealed_prefixes": prefixes,
        "archives": archives,
        "finished_epoch": time.time(),
        "original_deadline_epoch": deadline,
        "no_training_executed": True,
    }
    (output / "pack-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_blob_entries(paths, rows):
    """Keep every identity/path; store one regular blob per exact SHA, never hardlink."""
    entries, blobs, sizes = [], {}, {}
    for path in paths:
        row = rows[path]
        digest = row["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid-blob-sha")
        if digest in sizes and sizes[digest] != row["bytes"]:
            raise ValueError("same-sha-conflicting-size")
        sizes[digest] = row["bytes"]
        member = "blobs/" + digest
        entries.append({**row, "member": member})
        blobs.setdefault(member, Path(path))
    return entries, blobs


def restore_selected(archive_path, archive_sha, prefix, native, deadline):
    """Verify every blob; restore only one full native plus input/journal/source companions.

    All original absolute paths become prefix/<original-path-without-leading-slash>.
    No native/input bytes or relative paths change. No Torch load, training or generation.
    """
    guard(deadline)
    check_file(archive_path, archive_sha)
    if prefix.exists():
        raise ValueError("fresh-restore-prefix-required-no-overwrite")
    with tarfile.open(archive_path, "r|gz") as archive:
        first = archive.next()
        if not first.isreg() or first.name != "artifact-map.json" or first.size > 16 * BLOCK:
            raise ValueError("bounded-first-regular-artifact-map-required")
        ledger = json.loads(archive.extractfile(first).read())
        if (
            ledger["schema"] != "cpu-v3-content-addressed-regular-blob-map-v1"
            or ledger["source_commit"] != SOURCE
            or not ledger.get("source_bundle_sha256")
        ):
            raise ValueError("cpu-v3-source-bundle-map-required")
        rows = {}
        expected = {}
        for row in ledger["files"]:
            path = Path(row["path"])
            if (
                not path.is_absolute()
                or ".." in path.parts
                or "\\" in str(path)
                or row["path"] in rows
                or row["member"] != "blobs/" + row["sha256"]
            ):
                raise ValueError("unsafe-or-duplicate-original-path")
            rows[row["path"]] = row
            if row["member"] in expected and expected[row["member"]] != row["bytes"]:
                raise ValueError("same-blob-size-conflict")
            expected[row["member"]] = row["bytes"]
        manifest = ledger["native_manifests"].get(str(native))
        if manifest is None or manifest["source_commit"] != SOURCE:
            raise ValueError("selected-native-not-sealed")
        if manifest["schema"] != "torch-search-acting-native-cpu-v3":
            raise ValueError("selected-native-not-cpu-v3")
        selected = [str(native / "checkpoint.json")]
        selected += [str(native / name) for name in FILES]
        selected += [
            os.path.normpath(str(native / item["relative_path"]))
            for item in manifest["inputs"].values()
        ]
        epoch = manifest["state"]["epoch"]
        if epoch:
            selected.append(str(native.parent.parent / "journal" / f"epoch-{epoch:08d}.json.gz"))
        metadata = str(native.parent.parent / "metadata.json")
        if metadata in rows:
            selected.append(metadata)
        if any(path not in rows for path in selected):
            raise ValueError("selected-native-companions-incomplete")
        source_member = "blobs/" + ledger["source_bundle_sha256"]
        source_bytes = ledger["source_bundle_bytes"]
        if source_member in expected and expected[source_member] != source_bytes:
            raise ValueError("source-blob-size-conflict")
        expected[source_member] = source_bytes
        targets = {}
        for path in selected:
            restored = prefix / Path(path).relative_to("/")
            targets.setdefault(rows[path]["member"], []).append(restored)
        targets.setdefault(source_member, []).append(prefix / "source.bundle")
        required = sum(rows[p]["bytes"] for p in selected) + source_bytes + 128 * BLOCK
        if shutil.disk_usage(prefix.parent).free < required:
            raise ValueError("selected-restore-disk-headroom-insufficient")
        prefix.mkdir(parents=False, exist_ok=False)
        seen = set()
        for member in iter(archive.next, None):
            guard(deadline)
            if (
                not member.isreg()
                or member.name in seen
                or member.name not in expected
                or member.size != expected[member.name]
            ):
                raise ValueError("unexpected-link-duplicate-size-or-extra-member")
            destinations = targets.get(member.name, [])
            output = None
            if destinations:
                destinations[0].parent.mkdir(parents=True, exist_ok=True)
                output = destinations[0].open("xb")
            h = hashlib.sha256()
            try:
                with archive.extractfile(member) as stream:
                    while block := stream.read(BLOCK):
                        guard(deadline)
                        if shutil.disk_usage(prefix).free < 128 * BLOCK:
                            raise ValueError("restore-reserved-disk-headroom")
                        h.update(block)
                        if output is not None:
                            output.write(block)
            finally:
                if output is not None:
                    output.close()
            if h.hexdigest() != member.name.removeprefix("blobs/"):
                raise ValueError("regular-blob-sha-mismatch")
            for destination in destinations[1:]:
                guard(deadline)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destinations[0].open("rb") as src, destination.open("xb") as dst:
                    while block := src.read(BLOCK):
                        guard(deadline)
                        if shutil.disk_usage(prefix).free < 128 * BLOCK:
                            raise ValueError("restore-reserved-disk-headroom")
                        dst.write(block)
            seen.add(member.name)
        if seen != set(expected):
            raise ValueError("complete-blob-inventory-not-restored-verified")
    restored_rows = {}
    for path in selected:
        restored = prefix / Path(path).relative_to("/")
        check_file(restored, rows[path]["sha256"])
        restored_rows[str(restored)] = rows[path]
    restored_native = prefix / native.relative_to("/")
    if json.loads((restored_native / "checkpoint.json").read_text()) != manifest:
        raise ValueError("map-native-descriptor-not-original-manifest")
    check_native(restored_native, restored_rows)
    check_file(prefix / "source.bundle", ledger["source_bundle_sha256"])
    result = {
        "schema": "cpu-v3-selected-native-byte-restore-v1",
        "status": "bytes-pass",
        "archive_sha256": archive_sha,
        "original_native": str(native),
        "restored_native": str(restored_native),
        "verified_unique_blobs": len(seen),
        "restored_original_paths": len(selected),
        "source_commit": SOURCE,
        "no_training_or_generation": True,
        "native_runtime_loaded": False,
        "no_cuda_restore_claim": True,
        "original_deadline_epoch": deadline,
        "finished_epoch": time.time(),
    }
    (prefix / "byte-restore-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack")
    for name in ("spec", "prior-manifest", "output"):
        packing.add_argument("--" + name, type=Path, required=True)
    packing.add_argument("--spec-sha256", required=True)
    packing.add_argument("--deadline-epoch", type=float, required=True)
    restoring = commands.add_parser("restore")
    for name in ("archive", "prefix", "native"):
        restoring.add_argument("--" + name, type=Path, required=True)
    restoring.add_argument("--archive-sha256", required=True)
    restoring.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        result = pack(
            args.spec, args.spec_sha256, args.prior_manifest, args.output, args.deadline_epoch
        )
    else:
        result = restore_selected(
            args.archive, args.archive_sha256, args.prefix, args.native, args.deadline_epoch
        )
    print(json.dumps(result))
