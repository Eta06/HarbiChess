"""Package an approved source7 qualification inventory without altering its files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

SOURCE = "c022bc1605b44c3089439da5c6efb7bd4db4ff81"
MAX_TOTAL = 8 * 1024**3
MAX_ASSETS = 200
BLOCK = 1024 * 1024
END = 1791180000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(BLOCK):
            digest.update(block)
    return digest.hexdigest()


def guard(deadline):
    if time.time() >= min(deadline, END):
        raise RuntimeError("original-packaging-deadline-expired")


def read_verified(path, digest):
    if path.is_symlink() or not path.is_file() or sha(path) != digest:
        raise ValueError("approved-file-not-exact-regular")


def validate_inventory(obj):
    if obj["schema"] != "source7-closed-qualification-explicit-inventory-v1":
        raise ValueError("inventory-schema")
    if obj["source_commit"] != SOURCE or len(obj["natives"]) != 8:
        raise ValueError("inventory-source-or-native-count")
    paths = [f["path"] for f in obj["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate-inventory-path")
    roots = [Path(p) for p in obj["roots"]]
    extras = {
        "/content/harbichess-fullgame-method2-inputs/own-terminal-curriculum-preflight-20261005/book.json",
        "/content/harbichess-inputs/initial-e8.safetensors",
    }
    for item in obj["files"]:
        path = Path(item["path"])
        if ".." in path.parts or not path.is_absolute() or "\\" in str(path):
            raise ValueError("unsafe-original-path")
        if item["path"] not in extras and not any(path.is_relative_to(r) for r in roots):
            raise ValueError("outside-explicit-reviewed-roots")
    file_map = {x["path"]: x for x in obj["files"]}
    for native in obj["natives"]:
        if native["schema"] != "torch-search-acting-native-cuda-v2":
            raise ValueError("native-schema")
        for name, digest in native["artifacts"].items():
            payload = str(Path(native["path"]).parent / name)
            if payload not in file_map or file_map[payload]["sha256"] != digest:
                raise ValueError("native-artifact-inventory-mismatch")
        if set(native["inputs"]) != {"book", "experiment_config", "initial_weights", "protocol"}:
            raise ValueError("native-four-input-identity-required")
        for key, identity in native["inputs"].items():
            resolved = os.path.normpath(
                str(Path(native["path"]).parent / identity["relative_path"])
            )
            if (
                resolved not in file_map
                or file_map[resolved]["sha256"] != identity["sha256"]
                or native["run_config"]["input_sha256"][key] != identity["sha256"]
            ):
                raise ValueError("missing-or-changed-native-input-companion")
        state = native["state"]
        if state["replay_buffer"] != "closed-empty" or state["training_pass"] != "closed":
            raise ValueError("unclosed-native")
        if state["pending_search_schedule"] != "closed-empty":
            raise ValueError("pending-search-schedule")


def add_file(archive, name, path):
    info = tarfile.TarInfo(name)
    info.size = path.stat().st_size
    info.mode = 0o600
    with path.open("rb") as stream:
        archive.addfile(info, stream)


def pack(inventory_path, inventory_sha, approval_path, approval_sha, output, deadline):
    guard(deadline)
    read_verified(inventory_path, inventory_sha)
    read_verified(approval_path, approval_sha)
    obj = json.loads(inventory_path.read_text())
    approval = json.loads(approval_path.read_text())
    validate_inventory(obj)
    if (
        approval.get("status") != "approved-exact-public-byte-allowlist"
        or approval.get("inventory_sha256") != inventory_sha
        or approval.get("reviewed_public_sha256") != {x["path"]: x["sha256"] for x in obj["files"]}
    ):
        raise ValueError("explicit-exact-approval-required")
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(obj["source_repo"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != SOURCE:
        raise ValueError("source-pin")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise ValueError("source-dirty")
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
    entries = []
    for index, item in enumerate(obj["files"]):
        guard(deadline)
        path = Path(item["path"])
        read_verified(path, item["sha256"])
        entries.append({**item, "archive_member": f"payload/{index:06d}"})
    ledger = {
        "schema": "source7-public-qualified-and-failed-artifact-map-v1",
        "source_commit": SOURCE,
        "inventory_sha256": inventory_sha,
        "approval_sha256": approval_sha,
        "files": entries,
        "source_bundle": {"member": "source.bundle", "sha256": sha(bundle)},
        "claim_scope": "Original source7 profile failed; infrastructure-only byte persistence.",
    }
    ledger_bytes = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode()
    raw_path = output / "archive.pending.tar.gz"
    with raw_path.open("xb") as raw:
        with (
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as gz,
            tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as archive,
        ):
            info = tarfile.TarInfo("artifact-map.json")
            info.size = len(ledger_bytes)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(ledger_bytes))
            add_file(archive, "source.bundle", bundle)
            for item in entries:
                guard(deadline)
                add_file(archive, item["archive_member"], Path(item["path"]))
        raw.flush()
        os.fsync(raw.fileno())
    if raw_path.stat().st_size > 1024**3:
        raise ValueError("asset-over-1GiB")
    # Every member is streamed and hashed; no link extraction or blind tar trust.
    observed = {}
    with tarfile.open(raw_path, "r|gz") as archive:
        for member in archive:
            guard(deadline)
            if not member.isreg() or member.name in observed:
                raise ValueError("nonregular-or-duplicate-member")
            h = hashlib.sha256()
            stream = archive.extractfile(member)
            with stream:
                while block := stream.read(BLOCK):
                    h.update(block)
            observed[member.name] = h.hexdigest()
    expected = {i["archive_member"]: i["sha256"] for i in entries}
    expected.update(
        {
            "artifact-map.json": hashlib.sha256(ledger_bytes).hexdigest(),
            "source.bundle": sha(bundle),
        }
    )
    if observed != expected:
        raise ValueError("complete-archive-member-proof-mismatch")
    for item in entries:
        guard(deadline)
        read_verified(Path(item["path"]), item["sha256"])
    digest = sha(raw_path)
    final = output / f"sha256-{digest}.tar.gz"
    raw_path.rename(final)
    result = {
        "status": "closed-source7-explicit-archive-pack-pass",
        "archive": str(final),
        "sha256": digest,
        "bytes": final.stat().st_size,
        "member_count": len(observed),
        "source_commit": SOURCE,
        "inventory_sha256": inventory_sha,
        "original_deadline_epoch": deadline,
        "finished_epoch": time.time(),
        "full_public_readback_verified": False,
    }
    (output / "pack-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("inventory", "approval", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("inventory-sha256", "approval-sha256"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            pack(
                a.inventory,
                a.inventory_sha256,
                a.approval,
                a.approval_sha256,
                a.output,
                a.deadline_epoch,
            )
        )
    )
