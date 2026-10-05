"""Restore one public source7 archive and strict-load/save native E2; no updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

SOURCE = "c022bc1605b44c3089439da5c6efb7bd4db4ff81"
END = 1791180000
BLOCK = 1024 * 1024
NATIVE = (
    "content/harbichess-runs/certificate-source7-clean-CLI600-20261005/"
    "whole/checkpoints/epoch-00000002"
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while b := stream.read(BLOCK):
            h.update(b)
    return h.hexdigest()


def safe_relative(path):
    p = Path(path)
    if p.is_absolute() or ".." in p.parts or "\\" in path:
        raise ValueError("unsafe-archive-map-path")
    return p


def restore(archive_path, archive_sha, output, inventory_sha, deadline):
    def guard():
        if time.time() >= min(deadline, END):
            raise RuntimeError("original-restore-proof-deadline-expired")

    guard()
    if sha(archive_path) != archive_sha:
        raise ValueError("public-archive-sha-mismatch")
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r|gz") as archive:
        iterator = iter(archive)
        first = next(iterator)
        if not first.isreg() or first.name != "artifact-map.json" or first.size > 1024**2:
            raise ValueError("first-member-must-be-bounded-artifact-map")
        ledger = json.load(archive.extractfile(first))
        if (
            ledger["schema"] != "source7-public-qualified-and-failed-artifact-map-v1"
            or ledger["source_commit"] != SOURCE
            or ledger["inventory_sha256"] != inventory_sha
        ):
            raise ValueError("archive-ledger-source-inventory-mismatch")
        mapping = {}
        for row in ledger["files"]:
            original = row["path"]
            if not original.startswith("/content/"):
                raise ValueError("original-outside-content")
            member = str(safe_relative(row["archive_member"]))
            if member in mapping:
                raise ValueError("duplicate-mapped-member")
            mapping[member] = (output / safe_relative(original.lstrip("/")), row["sha256"])
        mapping["source.bundle"] = (output / "source.bundle", ledger["source_bundle"]["sha256"])
        observed = set()
        for member in iterator:
            guard()
            if not member.isreg() or member.name not in mapping or member.name in observed:
                raise ValueError("unexpected-link-or-member")
            target, digest = mapping[member.name]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as dest, archive.extractfile(member) as source:
                while b := source.read(BLOCK):
                    guard()
                    dest.write(b)
            if sha(target) != digest:
                raise ValueError("restored-member-sha-mismatch")
            observed.add(member.name)
        if observed != set(mapping):
            raise ValueError("missing-archive-members")
    guard()
    repo = output / "clean-source-c022"
    subprocess.run(
        ["git", "clone", "--quiet", str(output / "source.bundle"), str(repo)], check=True
    )
    subprocess.run(["git", "checkout", "--quiet", "--detach", SOURCE], cwd=repo, check=True)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise ValueError("restored-source-dirty")
    sys.path.insert(0, str(repo / "src"))
    import torch

    from harbichess.selfplay.online_actor import OnlineActorConfig
    from harbichess.training.ownsearch_targets import OwnSearchConfig
    from harbichess.training.torch_fullgame_ppo import FullGamePPOTrainConfig
    from harbichess.training.torch_ownsearch_core import OwnSearchObjective
    from harbichess.training.torch_search_acting_checkpoint import (
        load_search_acting_checkpoint,
        save_search_acting_checkpoint,
    )
    from harbichess.training.torch_search_acting_learner import (
        TorchSearchActingConfig,
        TorchSearchActingLearner,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("actual-CUDA-required-no-skip")
    native = output / NATIVE
    manifest = json.loads((native / "checkpoint.json").read_text())
    cfg = dict(manifest["run_config"]["config"])
    cfg["actors"] = OnlineActorConfig(**cfg["actors"])
    cfg["objective"] = OwnSearchObjective(**cfg["objective"])
    cfg["schedule"] = FullGamePPOTrainConfig(**cfg["schedule"])
    cfg["search"] = OwnSearchConfig(**cfg["search"])
    if cfg["device"] != "cuda:0":
        raise ValueError("actual-CUDA-native-required")
    inputs = {
        k: Path(os.path.normpath(str(native / v["relative_path"])))
        for k, v in manifest["inputs"].items()
    }
    guard()
    learner = TorchSearchActingLearner.fresh(
        config=TorchSearchActingConfig(**cfg), input_paths=inputs, source_commit=SOURCE
    )
    load_search_acting_checkpoint(native, learner)
    guard()
    # Same depth retains exact relative-input identity, without manifest rewriting.
    roundtrip = native.with_name("restore-proof-closed-roundtrip")
    save_search_acting_checkpoint(roundtrip, learner)
    files = ["checkpoint.json", *manifest["artifacts"]]
    matches = {name: sha(native / name) for name in files}
    if any(sha(roundtrip / name) != digest for name, digest in matches.items()):
        raise ValueError("strict-native-roundtrip-byte-mismatch")
    guard()
    result = {
        "schema": "actual-public-source7-CUDA-native-restore-proof-v1",
        "status": "pass",
        "archive_sha256": archive_sha,
        "source_commit": SOURCE,
        "native_epoch": 2,
        "all_six_payloads_plus_manifest_bytes_exact": matches,
        "all_original_input_paths_remapped_by_preserved_relative_geometry": True,
        "fresh_restored_repository": str(repo),
        "native": str(native),
        "training_updates_executed": 0,
        "original_profile_failure_preserved": True,
        "original_deadline_epoch": deadline,
        "finished_epoch": time.time(),
        "scope": (
            "Public-byte restore and complete strictCUDA model/Adam/RNG/cursor load/save; "
            "no next update or strength claim."
        ),
    }
    (output / "restore-proof-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("output", "public-readback-proof"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("asset-sha256", "inventory-sha256"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    proof = json.loads(a.public_readback_proof.read_text())
    if proof["status"] != "pass" or not proof["anonymous_full_body_size_sha256_verified"]:
        raise ValueError("actual-public-readback-proof-required")
    name = f"sha256-{a.asset_sha256}.tar.gz"
    rows = [x for x in proof["result"]["assets"] if x["name"] == name]
    if len(rows) != 1 or rows[0]["sha256"] != a.asset_sha256:
        raise ValueError("requested-asset-not-in-public-proof")
    # Fixed anonymous browser URL; no API token, caller URL, or authorization header.
    url = f"https://github.com/Eta06/HarbiChess/releases/download/port-linux-preflight-20261002/{name}"
    archive_path = a.output.with_suffix(".public-archive.tar.gz")
    request = urllib.request.Request(url, headers={"User-Agent": "HarbiChess-public-restore-proof"})
    with urllib.request.urlopen(request, timeout=30) as response, archive_path.open("xb") as dest:
        size = 0
        while b := response.read(BLOCK):
            if time.time() >= min(a.deadline_epoch, END):
                raise RuntimeError("original-restore-proof-deadline-expired")
            size += len(b)
            if size > rows[0]["bytes"]:
                raise ValueError("public-download-oversize")
            dest.write(b)
    if size != rows[0]["bytes"]:
        raise ValueError("public-download-size")
    print(
        json.dumps(
            restore(archive_path, a.asset_sha256, a.output, a.inventory_sha256, a.deadline_epoch)
        )
    )
