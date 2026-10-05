"""Package immutable fullgame closed epochs and serve only SHA-verified archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.server
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

RUNS = Path("/content/harbichess-runs")
RUN_TEMPLATE = "unconfigured"
REGISTERED_SEEDS = (20261425, 20261426, 20261525, 20261526)
JOB_SPEC = None
CURRENT_JOB = None
REGISTRATION_SCHEMA = None
REGISTRATION_SLOT = None
SOURCE_REPO = Path("/content/HarbiChess-fullgame-method2")
CONTENT_ROOT = Path("/content")
INPUT_ROOTS = (
    Path("/content/harbichess-inputs"),
    Path("/content/harbichess-fullgame-method2-inputs"),
)
REGISTRATION_NAME = "registration.json"
EXPECTED_PROTOCOL_SHA256 = "d2b8df68f0db92b1712dcfeba70a63d582b721270fb679e093d9f393ce25792c"
OUTPUT_ROOT = RUNS / "release-packages-formal45"
NATIVE_FILES = (
    "model.safetensors",
    "base.safetensors",
    "behavior.safetensors",
    "training.pt",
    "actor.json",
    "last-frozen-epoch.json.gz",
)
NATIVE_SCHEMAS = {"torch-fullgame-native-cuda-v1"}
RELEASE_ID = 401698693
RELEASE_TAG = "port-linux-preflight-20261002"
MAX_ASSET = 1024**3
MAX_TOTAL = 8 * 1024**3
MAX_PACKAGE_ASSETS = 100
MAX_ASSETS = 200
BLOCK = 1024 * 1024
HARD_STOP = datetime(2026, 10, 5, 6, 0, tzinfo=UTC).timestamp()
BASE_URL_RE = re.compile(r"https://[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com\Z")
ASSET_NAME_RE = re.compile(r"sha256-[0-9a-f]{64}\.tar\.gz\Z")


class BackupError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            h.update(block)
    return h.hexdigest()


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BackupError(f"not-regular-file:{path.name}")
    return path.resolve(strict=True)


def _inside(path: Path, roots: tuple[Path, ...]) -> Path:
    resolved = _regular_file(path)
    for root in roots:
        root_resolved = root.resolve(strict=True)
        if resolved.is_relative_to(root_resolved):
            return resolved
    raise BackupError(f"input-outside-approved-roots:{path.name}")


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BackupError(f"invalid-json:{path.name}") from exc
    if not isinstance(value, dict):
        raise BackupError(f"json-object-required:{path.name}")
    return value


def _verify_native(
    native: Path, epoch: int, expected_source: str | None = None
) -> tuple[dict, dict[str, Path]]:
    if native.is_symlink() or not native.is_dir():
        raise BackupError("native-checkpoint-not-directory")
    native = native.resolve(strict=True)
    _regular_file(native / "checkpoint.json")
    _regular_file(native / "actor.json")
    manifest = _json(native / "checkpoint.json")
    if manifest.get("schema") not in NATIVE_SCHEMAS:
        raise BackupError("native-schema-not-exact-own-method-cuda")
    source = manifest.get("source_commit")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise BackupError("native-source-commit-invalid")
    if CURRENT_JOB is not None and source != CURRENT_JOB["source_commit"]:
        raise BackupError("native-source-not-exact-method-job-pin")
    if expected_source is not None and source != expected_source:
        raise BackupError("native-source-commit-changed")
    state = manifest.get("state")
    if (
        not isinstance(state, dict)
        or state.get("epoch") != epoch
        or state.get("replay_buffer") != "closed-empty"
        or state.get("training_pass") != "closed"
        or state.get("pending_search_schedule") != "closed-empty"
    ):
        raise BackupError("native-boundary-not-closed-or-epoch-mismatch")
    actor = _json(native / "actor.json")
    if actor != state:
        raise BackupError("native-actor-state-differs-from-manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(NATIVE_FILES):
        raise BackupError("native-six-file-artifact-map-invalid")
    paths: dict[str, Path] = {}
    for name in NATIVE_FILES:
        path = _regular_file(native / name)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", artifacts[name])
            or sha256_file(path) != artifacts[name]
        ):
            raise BackupError(f"native-payload-hash-mismatch:{name}")
        paths[name] = path
    return manifest, paths


def _add_file(tar: tarfile.TarFile, arcname: str, source: Path) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = source.stat().st_size
    info.mode = 0o600
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    with source.open("rb") as stream:
        tar.addfile(info, stream)


def _add_hardlink(tar: tarfile.TarFile, arcname: str, target: str) -> None:
    info = tarfile.TarInfo(arcname)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    info.mode = 0o600
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.size = 0
    tar.addfile(info)


def _write_tar_gz(
    destination: Path, files: list[tuple[str, Path]], *, hardlinks: list[tuple[str, str]] = ()
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_upper_bound = sum(source.stat().st_size for _, source in files)
    if shutil.disk_usage(destination.parent).free < raw_upper_bound + 128 * 1024**2:
        raise BackupError("insufficient-disk-headroom-for-next-archive")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as raw:
        with (
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as gz,
            tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar,
        ):
            for arcname, source in files:
                if len(arcname.encode("utf-8")) > 100:
                    raise BackupError("archive-path-exceeds-ustar-limit")
                _add_file(tar, arcname, source)
            for arcname, target in hardlinks:
                if len(arcname.encode("utf-8")) > 100 or len(target.encode("utf-8")) > 100:
                    raise BackupError("hardlink-path-exceeds-ustar-limit")
                _add_hardlink(tar, arcname, target)
        raw.flush()
        os.fsync(raw.fileno())
    if destination.stat().st_size > MAX_ASSET:
        raise BackupError("single-archive-exceeds-1gib")


def _journal_record(path: Path, epoch: int) -> bytes:
    data = path.read_bytes()
    try:
        record = json.loads(gzip.decompress(data))
    except Exception as exc:
        raise BackupError(f"journal-gzip-or-json-invalid:{epoch}") from exc
    if not isinstance(record, dict) or record.get("epoch") != epoch:
        raise BackupError(f"journal-epoch-mismatch:{epoch}")
    return data


def _resolve_native_inputs(native0: Path, manifest: dict) -> dict[str, tuple[Path, str]]:
    inputs = manifest.get("inputs")
    required = {"initial_weights", "book", "experiment_config", "protocol"}
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise BackupError("native-input-map-not-exact-four-inputs")
    output = {}
    for key in sorted(required):
        item = inputs[key]
        if not isinstance(item, dict) or set(item) != {"relative_path", "sha256"}:
            raise BackupError(f"native-input-record-invalid:{key}")
        rel = item["relative_path"]
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." not in Path(rel).parts:
            raise BackupError(f"native-input-path-not-relative-outside-native:{key}")
        path = _inside(native0 / rel, INPUT_ROOTS)
        digest = item["sha256"]
        if not isinstance(digest, str) or sha256_file(path) != digest:
            raise BackupError(f"native-input-hash-mismatch:{key}")
        output[key] = (path, digest)
    return output


def _load_fixed_registration(
    input_records: dict[str, tuple[Path, str]], source_commit: str
) -> tuple[Path, bytes]:
    path, digest = input_records["protocol"]
    if digest != EXPECTED_PROTOCOL_SHA256 or sha256_file(path) != digest:
        raise BackupError("native-protocol-not-registered-formal-registration")
    data = path.read_bytes()
    try:
        registration = json.loads(data)
    except Exception as exc:
        raise BackupError("fixed-registration-json-invalid") from exc
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise BackupError("fixed-registration-schema-mismatch")
    if (
        registration.get("source_commit") != source_commit
        or registration.get("qualification_ledger_slot") != REGISTRATION_SLOT
        or registration.get("status") != "frozen-before-formal-execution"
    ):
        raise BackupError("fixed-registration-source-commit-mismatch")
    return path, data


def _assert_source_repo(repo: Path, source_commit: str) -> None:
    if (
        repo.is_symlink()
        or SOURCE_REPO.is_symlink()
        or repo.resolve(strict=True) != SOURCE_REPO.resolve(strict=True)
    ):
        raise BackupError("source-repo-not-fixed-registered-path")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != source_commit or dirty:
        raise BackupError("source-repo-not-exact-clean-native-commit")


def _source_bundle(repo: Path, source_commit: str, dest: Path) -> str:
    _assert_source_repo(repo, source_commit)
    subprocess.run(
        ["git", "bundle", "create", str(dest), "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    verify = subprocess.run(
        ["git", "bundle", "verify", str(dest)], cwd=repo, check=True, capture_output=True, text=True
    )
    heads = subprocess.run(
        ["git", "bundle", "list-heads", str(dest)],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_commit not in heads or not verify.stdout:
        raise BackupError("source-git-bundle-verification-failed")
    os.chmod(dest, 0o600)
    return sha256_file(dest)


def _validate_base_url(base_url: str) -> str:
    if not BASE_URL_RE.fullmatch(base_url):
        raise BackupError("base-url-must-be-https-single-label-trycloudflare-host-root")
    return base_url


KNOWN_METHODS = {
    4: {
        "seeds": (20261425, 20261426),
        "source": "a278bba67bce962cb9294f0d24040e02e9acf1f4",
        "native_schema": "torch-ownsearch-native-cuda-v1",
        "registration_schema": "ufuk-ownsearch-method4-prospective-v1",
        "run_template": "ownsearch-method4-seed-{seed}/run",
    },
    5: {
        "seeds": (20261525, 20261526),
        "source": "4515a7c0dda3b4f9615c2fc78a47c872ab14699d",
        "native_schema": "torch-search-acting-native-cuda-v2",
        "registration_schema": "ufuk-search-acting-method5-prospective-v1",
        "run_template": "search-acting-method5-seed-{seed}/run",
    },
}


def load_spec(path: Path, digest: str) -> None:
    global JOB_SPEC
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256_file(_regular_file(path)) != digest:
        raise BackupError("extension-spec-sha256-mismatch")
    spec = _json(path)
    if (
        spec.get("schema") != "formal45-release-backup-spec-v1"
        or spec.get("status") != "frozen-reviewed-artifact-allowlist"
    ):
        raise BackupError("extension-spec-schema-mismatch")
    if set(spec.get("jobs", {})) != set(map(str, REGISTERED_SEEDS)):
        raise BackupError("extension-spec-exact-four-seeds-required")
    for seed, job in spec["jobs"].items():
        method = KNOWN_METHODS.get(job.get("slot"))
        if not method or int(seed) not in method["seeds"]:
            raise BackupError("extension-job-method-seed-mismatch")
        if job.get("source_commit") != method["source"]:
            raise BackupError("extension-job-source-pin-mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", job.get("protocol_sha256", "")):
            raise BackupError("extension-job-protocol-sha256-required")
        repo = Path(job["source_repo"])
        if repo.is_symlink() or not repo.resolve(strict=True).is_relative_to(CONTENT_ROOT):
            raise BackupError("extension-source-repo-outside-content")
        if (
            not isinstance(job.get("reviewed_public_evidence"), list)
            or not job["reviewed_public_evidence"]
        ):
            raise BackupError("extension-evidence-list-required")
    spec["verified_spec_sha256"] = digest
    JOB_SPEC = spec


def configure_seed(seed: int) -> None:
    global CURRENT_JOB, RUN_TEMPLATE, SOURCE_REPO, NATIVE_SCHEMAS
    global EXPECTED_PROTOCOL_SHA256, REGISTRATION_SCHEMA, REGISTRATION_SLOT
    if JOB_SPEC is None or seed not in REGISTERED_SEEDS:
        raise BackupError("extension-spec-required-before-pack")
    CURRENT_JOB = JOB_SPEC["jobs"][str(seed)]
    method = KNOWN_METHODS[CURRENT_JOB["slot"]]
    RUN_TEMPLATE = method["run_template"]
    SOURCE_REPO = Path(CURRENT_JOB["source_repo"])
    NATIVE_SCHEMAS = {method["native_schema"]}
    EXPECTED_PROTOCOL_SHA256 = CURRENT_JOB["protocol_sha256"]
    REGISTRATION_SCHEMA = method["registration_schema"]
    REGISTRATION_SLOT = CURRENT_JOB["slot"]


def verified_evidence(job: dict) -> list[dict]:
    """Only individually reviewed immutable SHA-pinned regular files, never recursion."""
    rows = []
    seen = set()
    for index, entry in enumerate(job["reviewed_public_evidence"]):
        if set(entry) != {"path", "sha256", "purpose", "reviewed_public"}:
            raise BackupError("evidence-record-fields-not-exact")
        if entry["reviewed_public"] is not True or not isinstance(entry["purpose"], str):
            raise BackupError("evidence-not-explicitly-reviewed-public")
        path = _inside(Path(entry["path"]), (RUNS, *INPUT_ROOTS))
        if path in seen:
            raise BackupError("duplicate-evidence-path")
        seen.add(path)
        digest = entry["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BackupError("evidence-sha256-required")
        if sha256_file(path) != digest:
            raise BackupError("evidence-file-sha256-mismatch")
        rows.append(
            {
                "original_path": str(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "purpose": entry["purpose"],
                "archive_path": f"metadata/evidence/{index:06d}.bin",
                "resolved": path,
            }
        )
    return rows


def verify_archive(path: Path) -> dict[str, dict]:
    """Stream-check regular members and the single supported native journal alias.

    Does not extract or follow filesystem links. Unknown link types are rejected.
    """
    records = {}
    total = 0
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            name = member.name
            parts = Path(name).parts
            if (
                not parts
                or Path(name).is_absolute()
                or ".." in parts
                or name in records
                or "\\" in name
            ):
                raise BackupError("unsafe-or-duplicate-archive-member")
            if member.islnk():
                match = re.fullmatch(
                    r"checkpoints/epoch-([0-9]{8})/last-frozen-epoch\.json\.gz", name
                )
                target = member.linkname
                if (
                    not match
                    or target != f"journal/epoch-{match.group(1)}.json.gz"
                    or target not in records
                    or records[target]["kind"] != "regular"
                ):
                    raise BackupError("unsupported-or-unresolved-archive-hardlink")
                records[name] = {
                    **records[target],
                    "kind": "verified-journal-alias",
                    "target": target,
                }
                continue
            if not member.isreg():
                raise BackupError("nonregular-archive-member")
            total += member.size
            if total > MAX_TOTAL:
                raise BackupError("expanded-archive-size-over-limit")
            stream = archive.extractfile(member)
            if stream is None:
                raise BackupError("archive-member-cannot-read")
            digest, size = hashlib.sha256(), 0
            with stream:
                while block := stream.read(BLOCK):
                    digest.update(block)
                    size += len(block)
            if size != member.size:
                raise BackupError("archive-member-length-mismatch")
            records[name] = {"kind": "regular", "bytes": size, "sha256": digest.hexdigest()}
    return records


def pack_seed(seed: int, *, base_url: str) -> Path:
    configure_seed(seed)
    if seed not in REGISTERED_SEEDS:
        raise BackupError("seed-not-registered")
    base_url = _validate_base_url(base_url)
    run_root = RUNS / RUN_TEMPLATE.format(seed=seed)
    if (
        run_root.is_symlink()
        or not run_root.is_dir()
        or not run_root.resolve(strict=True).is_relative_to(RUNS.resolve(strict=True))
    ):
        raise BackupError("fixed-seed-run-root-missing-or-symlink")
    native_root = run_root / "checkpoints"
    journal_root = run_root / "journal"
    dirs = []
    for path in native_root.iterdir():
        match = re.fullmatch(r"epoch-(\d{8})", path.name)
        if match:
            if path.is_symlink() or not path.is_dir():
                raise BackupError(f"malformed-checkpoint-entry:{path.name}")
            dirs.append((int(match.group(1)), path))
    dirs.sort()
    if not dirs or [epoch for epoch, _ in dirs] != list(range(dirs[-1][0] + 1)):
        raise BackupError("checkpoint-prefix-not-contiguous-from-zero")
    max_epoch = dirs[-1][0]
    if max_epoch < 1 or max_epoch + 3 > MAX_PACKAGE_ASSETS:
        raise BackupError("checkpoint-count-out-of-range-for-asset-cap")

    verified = []
    original_run_metadata = _regular_file(run_root / "metadata.json")
    original_run_metadata_sha = sha256_file(original_run_metadata)
    source_commit = None
    baseline_manifest = None
    for epoch, directory in dirs:
        manifest, paths = _verify_native(directory, epoch, source_commit)
        source_commit = manifest["source_commit"]
        run_config = manifest.get("run_config")
        config = run_config.get("config") if isinstance(run_config, dict) else None
        if not isinstance(config, dict) or config.get("seed") != seed:
            raise BackupError("native-run-config-seed-mismatch")
        if baseline_manifest is None:
            baseline_manifest = manifest
        elif (
            manifest.get("inputs") != baseline_manifest.get("inputs")
            or manifest.get("run_config") != baseline_manifest.get("run_config")
            or manifest.get("base_model_sha256") != baseline_manifest.get("base_model_sha256")
        ):
            raise BackupError("native-input-or-run-config-changed-between-epochs")
        if epoch == 0:
            if paths["last-frozen-epoch.json.gz"].stat().st_size != 0:
                raise BackupError("epoch-zero-native-must-have-empty-last-journal")
            journal = None
        else:
            journal_path = journal_root / f"epoch-{epoch:08d}.json.gz"
            _regular_file(journal_path)
            journal_data = _journal_record(journal_path, epoch)
            record = json.loads(gzip.decompress(journal_data))
            if record.get("schema") != manifest["run_config"].get("schema"):
                raise BackupError("journal-own-method-schema-mismatch")
            if record.get("previous_sample_chain_sha256") != verified[-1][2]["state"].get(
                "sample_chain_sha256"
            ):
                raise BackupError("journal-native-prefix-chain-discontinuity")
            if journal_data != paths["last-frozen-epoch.json.gz"].read_bytes() or record.get(
                "sample_chain_sha256"
            ) != manifest["state"].get("sample_chain_sha256"):
                raise BackupError(f"native-last-frozen-journal-differs:{epoch}")
            journal = journal_path
        verified.append((epoch, directory, manifest, paths, journal))

    frozen_native_inventory = {
        str(directory / name): sha256_file(directory / name)
        for _, directory, _, _, _ in verified
        for name in ("checkpoint.json", *NATIVE_FILES)
    }
    input_records = _resolve_native_inputs(dirs[0][1], verified[0][2])
    repo = SOURCE_REPO
    reg_path, reg_data = _load_fixed_registration(input_records, source_commit)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.resolve().is_relative_to(
        RUNS.resolve(strict=True)
    ):
        raise BackupError("output-root-not-a-real-subdirectory-of-runs")
    if OUTPUT_ROOT.resolve() == run_root.resolve() or OUTPUT_ROOT.resolve().is_relative_to(
        run_root.resolve()
    ):
        raise BackupError("output-root-overlaps-working-run")
    temp_root = Path(tempfile.mkdtemp(prefix=f".seed-{seed}-backup-", dir=OUTPUT_ROOT))
    os.chmod(temp_root, 0o700)
    archives = []
    try:
        for epoch, directory, _manifest, paths, journal in verified:
            name = f"epoch-{epoch:08d}"
            regular = []
            links = []
            if journal is not None:
                jrel = f"journal/epoch-{epoch:08d}.json.gz"
                regular.append((jrel, journal))
            else:
                jrel = None
            regular.append((f"checkpoints/{name}/checkpoint.json", directory / "checkpoint.json"))
            for file_name in NATIVE_FILES:
                if file_name == "last-frozen-epoch.json.gz" and journal is not None:
                    links.append((f"checkpoints/{name}/{file_name}", jrel))
                else:
                    regular.append((f"checkpoints/{name}/{file_name}", paths[file_name]))
            tar_path = temp_root / f"pending-{name}.tar.gz"
            _write_tar_gz(tar_path, regular, hardlinks=links)
            verify_archive(tar_path)
            digest = sha256_file(tar_path)
            asset_name = f"sha256-{digest}.tar.gz"
            final_tar = temp_root / asset_name
            os.rename(tar_path, final_tar)
            os.chmod(final_tar, 0o600)
            archives.append(
                {
                    "epoch": epoch,
                    "name": asset_name,
                    "bytes": final_tar.stat().st_size,
                    "sha256": digest,
                    "path": asset_name,
                }
            )

        bundle_path = temp_root / "source.bundle"
        bundle_sha = _source_bundle(repo, source_commit, bundle_path)
        if bundle_path.stat().st_size > MAX_ASSET:
            raise BackupError("source-git-bundle-exceeds-1gib")
        evidence = verified_evidence(CURRENT_JOB)
        metadata = {
            "schema": "harbichess-a100-run-metadata-v1",
            "seed": seed,
            "max_closed_epoch": max_epoch,
            "source_commit": source_commit,
            "source_bundle_sha256": bundle_sha,
            "original_run_metadata_sha256": original_run_metadata_sha,
            "backup_extension_spec_sha256": JOB_SPEC["verified_spec_sha256"],
            "backup_extension_helper_sha256": sha256_file(Path(__file__)),
            "reviewed_public_evidence": [
                {k: v for k, v in item.items() if k != "resolved"} for item in evidence
            ],
            "registration_sha256": hashlib.sha256(reg_data).hexdigest(),
            "inputs": {
                k: {
                    "path": str(p),
                    "basename": p.name,
                    "bytes": p.stat().st_size,
                    "sha256": d,
                    "archive_path": f"metadata/inputs/{k}.bin",
                }
                for k, (p, d) in sorted(input_records.items())
            },
            "journal_prefix": [
                {"epoch": epoch, "sha256": sha256_file(journal)}
                for epoch, _, _, _, journal in verified
                if journal is not None
            ],
            "checkpoint_epochs": [epoch for epoch, *_ in verified],
            "epoch_archives": [a for a in archives if a["epoch"] is not None],
        }
        metadata_path = temp_root / "run-metadata.json"
        metadata_path.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")
        os.chmod(metadata_path, 0o600)
        restore_text = temp_root / "restore.txt"
        restore_text.write_text(
            "Extract each epoch archive into the run directory in ascending epoch order; "
            "they contain native checkpoint directories and the complete journal prefix.\n"
            "The metadata archive contains source.bundle plus the exact registration and "
            "original input files under metadata/inputs/{key}.bin. Restore each to the exact "
            "metadata.inputs[key].path after verifying its recorded SHA-256 "
            "before restoring files to its recorded /content path.\n"
            "The source bundle is for the frozen source commit only; it does not contain "
            "runtime credentials, logs, or Git configuration.\n"
        )
        os.chmod(restore_text, 0o600)
        metadata_files = [
            ("metadata/source.bundle", bundle_path),
            ("metadata/original-run-metadata.json", original_run_metadata),
            (f"metadata/registration/{REGISTRATION_NAME}", reg_path),
            ("metadata/run-metadata.json", metadata_path),
            ("metadata/restore.txt", restore_text),
        ]
        for key, (path, _) in sorted(input_records.items()):
            # Original restore paths stay in metadata.inputs; avoid USTAR long paths.
            metadata_files.append((f"metadata/inputs/{key}.bin", path))
        for item in evidence:
            metadata_files.append((item["archive_path"], item["resolved"]))
        meta_tar = temp_root / "pending-metadata.tar.gz"
        _write_tar_gz(meta_tar, metadata_files)
        verify_archive(meta_tar)
        meta_sha = sha256_file(meta_tar)
        meta_name = f"sha256-{meta_sha}.tar.gz"
        meta_final = temp_root / meta_name
        os.rename(meta_tar, meta_final)
        os.chmod(meta_final, 0o600)
        archives.append(
            {
                "epoch": None,
                "name": meta_name,
                "bytes": meta_final.stat().st_size,
                "sha256": meta_sha,
                "path": meta_name,
            }
        )

        if len(archives) > MAX_ASSETS or sum(a["bytes"] for a in archives) > MAX_TOTAL:
            raise BackupError("archive-count-or-total-exceeds-registered-cap")
        package = {
            "schema": "harbichess-a100-release-package-v1",
            "seed": seed,
            "source_commit": source_commit,
            "max_closed_epoch": max_epoch,
            "archives": sorted(archives, key=lambda a: (a["epoch"] is None, a["epoch"] or 0)),
            "total_bytes": sum(a["bytes"] for a in archives),
        }
        package_bytes = (json.dumps(package, sort_keys=True, separators=(",", ":")) + "\n").encode()
        package_sha = hashlib.sha256(package_bytes).hexdigest()
        package_path = temp_root / "package.json"
        package_path.write_bytes(package_bytes)
        os.chmod(package_path, 0o600)
        transport = {
            "schema": "harbichess-a100-release-transport-v1",
            "release_id": RELEASE_ID,
            "release_tag": RELEASE_TAG,
            "seed": seed,
            "source_commit": source_commit,
            "package_manifest_sha256": package_sha,
            "assets": [
                {
                    "name": a["name"],
                    "bytes": a["bytes"],
                    "sha256": a["sha256"],
                    "url": f"{base_url}/{a['name']}",
                }
                for a in package["archives"]
            ],
        }
        transport_path = temp_root / "transport-manifest.json"
        transport_path.write_text(json.dumps(transport, indent=2, sort_keys=True) + "\n")
        os.chmod(transport_path, 0o600)

        if sha256_file(original_run_metadata) != original_run_metadata_sha:
            raise BackupError("original-run-metadata-changed-during-package")
        if verified_evidence(CURRENT_JOB) != evidence:
            raise BackupError("reviewed-evidence-changed-during-package")
        if any(
            sha256_file(Path(path)) != digest for path, digest in frozen_native_inventory.items()
        ):
            raise BackupError("closed-native-bytes-changed-during-package")
        # Recheck frozen natives, complete journals, original inputs and source bundle.
        for epoch, directory, _manifest, paths, journal in verified:
            _verify_native(directory, epoch, source_commit)
            if journal is not None and sha256_file(journal) != sha256_file(
                paths["last-frozen-epoch.json.gz"]
            ):
                raise BackupError(f"journal-changed-during-package:{epoch}")
        for key, (path, digest) in input_records.items():
            if sha256_file(path) != digest:
                raise BackupError(f"input-changed-during-package:{key}")
        if (
            hashlib.sha256(reg_path.read_bytes()).hexdigest()
            != hashlib.sha256(reg_data).hexdigest()
        ):
            raise BackupError("registration-changed-during-package")
        _assert_source_repo(repo, source_commit)
        if sha256_file(bundle_path) != bundle_sha:
            raise BackupError("source-bundle-changed-during-package")
        snapshot_digest = hashlib.sha256(package_bytes).hexdigest()
        final = OUTPUT_ROOT / f"seed-{seed}-through-{max_epoch:08d}-{snapshot_digest[:12]}"
        if final.exists():
            if final.is_symlink() or (final / "package.json").read_bytes() != package_bytes:
                raise BackupError("immutable-package-output-conflict")
            build_server_map(final)
            shutil.rmtree(temp_root)
            return final
        os.rename(temp_root, final)
        descriptor = os.open(OUTPUT_ROOT, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return final
    except BaseException:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise


def build_server_map(package_dir: Path) -> dict[str, tuple[Path, int, str, int, int]]:
    package_dir = package_dir.resolve(strict=True)
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise BackupError("package-root-invalid")
    manifest_path = _regular_file(package_dir / "package.json")
    package = _json(manifest_path)
    if package.get("schema") != "harbichess-a100-release-package-v1":
        raise BackupError("package-manifest-schema-invalid")
    assets = package.get("archives")
    if not isinstance(assets, list) or not 1 <= len(assets) <= MAX_PACKAGE_ASSETS:
        raise BackupError("package-asset-list-invalid")
    mapping = {}
    total = 0
    for item in assets:
        if not isinstance(item, dict):
            raise BackupError("package-asset-entry-invalid")
        name, rel, size, digest = (item.get(k) for k in ("name", "path", "bytes", "sha256"))
        if (
            not isinstance(name, str)
            or not ASSET_NAME_RE.fullmatch(name)
            or rel != name
            or type(size) is not int
            or not 0 < size <= MAX_ASSET
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or name in mapping
        ):
            raise BackupError("package-asset-fields-invalid")
        path = _regular_file(package_dir / name)
        if path.parent != package_dir:
            raise BackupError("package-asset-path-escapes-root")
        st = path.stat()
        if stat.S_IMODE(st.st_mode) != 0o600 or st.st_size != size:
            raise BackupError("package-asset-mode-or-size-invalid")
        if sha256_file(path) != digest or name != f"sha256-{digest}.tar.gz":
            raise BackupError("package-asset-sha-name-or-content-invalid")
        mapping["/" + name] = (path, size, digest, st.st_dev, st.st_ino)
        total += size
    if total > MAX_TOTAL:
        raise BackupError("package-asset-total-exceeds-cap")
    return mapping


class AllowlistHTTPServer(http.server.HTTPServer):
    allow_reuse_address = False
    daemon_threads = False

    def __init__(self, server_address, handler, files, deadline):
        self.files = files
        self.deadline = deadline
        self.timeout = 1.0
        super().__init__(server_address, handler)


class ReadOnlyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "HarbiChess-ReadOnly/1"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(max(0.1, min(2.0, self.server.deadline - time.time())))

    def log_message(self, fmt, *args):
        # Do not log URL/query/host or public tunnel addresses.
        return

    def _respond(self, status: int, body: bytes = b""):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        if status == 405:
            self.send_header("Allow", "GET, HEAD")
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)
        self.close_connection = True

    def _file(self, head: bool):
        if (
            self.path != urllib.parse.urlsplit(self.path).path
            or "?" in self.path
            or "%" in self.path
            or "\\" in self.path
        ):
            return self._respond(404)
        item = self.server.files.get(self.path)
        if item is None or time.time() >= self.server.deadline:
            return self._respond(404)
        path, size, digest, dev, ino = item
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            st = os.fstat(fd)
            if (
                not stat.S_ISREG(st.st_mode)
                or st.st_dev != dev
                or st.st_ino != ino
                or st.st_size != size
                or stat.S_IMODE(st.st_mode) != 0o600
            ):
                os.close(fd)
                return self._respond(503)
            stream = os.fdopen(fd, "rb")
        except OSError:
            return self._respond(503)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("ETag", f'"sha256-{digest}"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        if not head:
            try:
                while time.time() < self.server.deadline and (block := stream.read(BLOCK)):
                    self.connection.settimeout(
                        max(0.1, min(2.0, self.server.deadline - time.time()))
                    )
                    self.wfile.write(block)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        stream.close()
        self.close_connection = True

    def do_GET(self):
        self._file(head=False)

    def do_HEAD(self):
        self._file(head=True)

    def do_POST(self):
        self._respond(405, b"method not allowed\n")

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST
    do_OPTIONS = do_POST


def build_transport_manifest(package_dirs: list[Path], base_url: str) -> dict:
    base_url = _validate_base_url(base_url)
    archive_map = {}
    seeds = set()
    if JOB_SPEC is None:
        raise BackupError("manifest-spec-required")
    priors = JOB_SPEC.get("prior_transport_manifests")
    if not isinstance(priors, list) or not priors:
        raise BackupError("prior-public-asset-inventory-required")
    for previous in priors:
        path = _inside(Path(previous["path"]), INPUT_ROOTS)
        if sha256_file(path) != previous["sha256"]:
            raise BackupError("prior-public-manifest-sha256-mismatch")
        prior = _json(path)
        if (
            prior.get("schema") != "harbichess-a100-release-transport-v1"
            or prior.get("release_id") != RELEASE_ID
            or prior.get("release_tag") != RELEASE_TAG
        ):
            raise BackupError("prior-public-manifest-identity-mismatch")
        seeds.update(prior.get("seeds", []))
        for item in prior["assets"]:
            if (
                set(item) != {"name", "bytes", "sha256", "url"}
                or not ASSET_NAME_RE.fullmatch(item["name"])
                or item["name"] != f"sha256-{item['sha256']}.tar.gz"
                or type(item["bytes"]) is not int
                or not 0 < item["bytes"] <= MAX_ASSET
            ):
                raise BackupError("prior-public-manifest-asset-invalid")
            url = urllib.parse.urlsplit(item["url"])
            _validate_base_url(f"{url.scheme}://{url.netloc}")
            if url.path != "/" + item["name"] or url.query or url.fragment:
                raise BackupError("prior-public-manifest-asset-url-invalid")
            old = archive_map.get(item["name"])
            if old and (old["sha256"] != item["sha256"] or old["bytes"] != item["bytes"]):
                raise BackupError("prior-public-asset-conflict")
            archive_map[item["name"]] = item
    for package_dir in package_dirs:
        resolved = package_dir.resolve(strict=True)
        if not resolved.is_relative_to(OUTPUT_ROOT.resolve(strict=True)):
            raise BackupError("manifest-package-must-be-under-fixed-release-packages-root")
        package = _json(resolved / "package.json")
        build_server_map(resolved)
        seed = package.get("seed")
        if seed not in REGISTERED_SEEDS:
            raise BackupError("package-seed-not-registered")
        seeds.add(seed)
        for item in package.get("archives", []):
            name = item.get("name")
            if name in archive_map and (
                archive_map[name]["sha256"] != item.get("sha256")
                or archive_map[name]["bytes"] != item.get("bytes")
            ):
                raise BackupError("same-sha-asset-name-has-conflicting-metadata")
            archive_map[name] = item
    if len(archive_map) > MAX_ASSETS or not archive_map:
        raise BackupError("combined-asset-count-out-of-range")
    total = sum(item["bytes"] for item in archive_map.values())
    if total > MAX_TOTAL:
        raise BackupError("combined-total-exceeds-8gib")
    return {
        "schema": "harbichess-a100-release-transport-v1",
        "release_id": RELEASE_ID,
        "release_tag": RELEASE_TAG,
        "seeds": sorted(seed for seed in seeds if type(seed) is int),
        "assets": [
            {
                "name": item["name"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "url": item.get("url", f"{base_url}/{item['name']}"),
            }
            for item in sorted(archive_map.values(), key=lambda row: row["name"])
        ],
    }


def merge_server_maps(package_dirs: list[Path]) -> dict[str, tuple[Path, int, str, int, int]]:
    files = {}
    for package_dir in package_dirs:
        resolved = package_dir.resolve(strict=True)
        if not resolved.is_relative_to(OUTPUT_ROOT.resolve(strict=True)):
            raise BackupError("server-package-must-be-under-fixed-release-packages-root")
        for path, record in build_server_map(resolved).items():
            previous = files.get(path)
            if previous is not None and previous[1:3] != record[1:3]:
                raise BackupError("server-duplicate-asset-path-conflicts")
            files[path] = previous or record
    if not files:
        raise BackupError("server-has-no-allowlisted-files")
    if len(files) > MAX_ASSETS:
        raise BackupError("server-asset-count-exceeds-200")
    return files


def serve(
    package_dirs: list[Path], *, deadline: float, bind: str = "127.0.0.1", port: int = 18088
) -> None:
    if bind != "127.0.0.1" or port != 18088:
        raise BackupError("server-must-bind-only-loopback-127.0.0.1:18088")
    if not time.time() < deadline <= HARD_STOP:
        raise BackupError("server-expiry-must-be-future-and-no-later-than-2026-10-05T06:00Z")
    files = merge_server_maps(package_dirs)
    server = AllowlistHTTPServer((bind, port), ReadOnlyHandler, files, deadline)
    try:
        while time.time() < deadline:
            server.handle_request()
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    p_pack = sub.add_parser("pack")
    p_pack.add_argument("--seed", type=int, choices=REGISTERED_SEEDS, required=True)
    p_pack.add_argument("--base-url", required=True)
    p_server = sub.add_parser("serve")
    p_server.add_argument("--package-dir", type=Path, action="append", required=True)
    p_server.add_argument("--deadline-epoch", type=float, required=True)
    p_manifest = sub.add_parser("manifest")
    p_manifest.add_argument("--package-dir", type=Path, action="append", required=True)
    p_manifest.add_argument("--base-url", required=True)
    p_manifest.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_spec(args.spec, args.spec_sha256)
    if args.command == "pack":
        out = pack_seed(args.seed, base_url=args.base_url)
        package_sha = sha256_file(out / "package.json")
        print(
            json.dumps(
                {
                    "status": "package-ready",
                    "package_dir": str(out),
                    "seed": args.seed,
                    "package_manifest_sha256": package_sha,
                    "package_asset_count": len(_json(out / "package.json")["archives"]),
                    "scope": "closed native checkpoints and explicit registered inputs only",
                },
                sort_keys=True,
            )
        )
    elif args.command == "serve":
        serve(args.package_dir, deadline=args.deadline_epoch)
        print("readonly-allowlist-server-stopped")
    else:
        transport = build_transport_manifest(args.package_dir, args.base_url)
        output = args.output
        parent = output.parent.resolve(strict=True)
        if not parent.is_relative_to(OUTPUT_ROOT.resolve(strict=True)) or output.exists():
            raise BackupError("manifest-output-must-be-new-under-fixed-package-root")
        data = (json.dumps(transport, sort_keys=True, indent=2) + "\n").encode()
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        print(
            json.dumps(
                {
                    "status": "transport-manifest-ready",
                    "seeds": transport["seeds"],
                    "asset_count": len(transport["assets"]),
                    "manifest_sha256": hashlib.sha256(data).hexdigest(),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BackupError as exc:
        print(
            json.dumps({"status": "failed-preserved", "reason": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("readonly-allowlist-server-stopped-by-operator")
        sys.exit(0)
    except Exception:
        print(
            json.dumps(
                {"status": "failed-preserved", "reason": "unexpected-error-details-suppressed"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        sys.exit(1)
