"""Atomic checksummed gzip replay shards with strict compatibility checks."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harbichess.chess.rules import PythonChessRules
from harbichess.replay.schema import (
    SCHEMA_VERSIONS,
    SUPPORTED_TARGET_SCHEMA_VERSIONS,
    ReplayRecord,
)
from harbichess.replay.split import ReplaySplit


class ReplayCompatibilityError(ValueError):
    pass


class ReplayIntegrityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ShardMetadata:
    run_id: str
    generation: int
    source_checkpoint: str
    source_commit: str
    created_at: str
    split: ReplaySplit

    def __post_init__(self) -> None:
        if not self.run_id or not self.source_checkpoint or self.generation < 0:
            raise ValueError("shard identifiers must be present and generation non-negative")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit.lower()
        ):
            raise ValueError("source_commit must be a full Git SHA")


@dataclass(frozen=True, slots=True)
class ReplayShardHeader:
    replay_schema: int
    encoder_schema: int
    action_schema: int
    target_schema: int
    run_id: str
    generation: int
    source_checkpoint: str
    source_commit: str
    created_at: str
    split: ReplaySplit
    game_count: int
    sample_count: int
    payload_sha256: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayShardHeader:
        parsed = dict(data)
        parsed["split"] = ReplaySplit(parsed["split"])
        return cls(**parsed)


@dataclass(frozen=True, slots=True)
class ReplayShard:
    header: ReplayShardHeader
    records: tuple[ReplayRecord, ...]


def _canonical_line(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def write_shard_atomic(
    path: Path,
    records: Sequence[ReplayRecord],
    metadata: ShardMetadata,
) -> ReplayShardHeader:
    if not records:
        raise ValueError("cannot write an empty replay shard")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_fd, raw_name = tempfile.mkstemp(prefix=f".{path.name}.raw.", dir=path.parent)
    os.close(raw_fd)
    output_fd, output_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(output_fd)
    try:
        digest = hashlib.sha256()
        with open(raw_name, "wb") as raw:
            for record in records:
                line = _canonical_line(record.to_dict())
                raw.write(line)
                digest.update(line)
            raw.flush()
            os.fsync(raw.fileno())
        header = ReplayShardHeader(
            replay_schema=SCHEMA_VERSIONS["replay"],
            encoder_schema=SCHEMA_VERSIONS["encoder"],
            action_schema=SCHEMA_VERSIONS["action"],
            target_schema=SCHEMA_VERSIONS["target"],
            run_id=metadata.run_id,
            generation=metadata.generation,
            source_checkpoint=metadata.source_checkpoint,
            source_commit=metadata.source_commit,
            created_at=metadata.created_at,
            split=metadata.split,
            game_count=len({record.game_id for record in records}),
            sample_count=len(records),
            payload_sha256=digest.hexdigest(),
        )
        with open(output_name, "wb") as output:
            with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
                compressed.write(_canonical_line(asdict(header)))
                with open(raw_name, "rb") as raw:
                    shutil.copyfileobj(raw, compressed)
            output.flush()
            os.fsync(output.fileno())
        os.replace(output_name, path)
        return header
    finally:
        for temporary in (raw_name, output_name):
            if os.path.exists(temporary):
                os.unlink(temporary)


def read_shard(path: Path, *, rules: PythonChessRules | None = None) -> ReplayShard:
    engine = rules or PythonChessRules()
    try:
        with gzip.open(path, "rb") as compressed:
            header = ReplayShardHeader.from_dict(json.loads(compressed.readline()))
            actual_versions = {
                "replay": header.replay_schema,
                "encoder": header.encoder_schema,
                "action": header.action_schema,
                "target": header.target_schema,
            }
            base_versions_match = all(
                actual_versions[name] == SCHEMA_VERSIONS[name]
                for name in ("replay", "encoder", "action")
            )
            if (
                not base_versions_match
                or header.target_schema not in SUPPORTED_TARGET_SCHEMA_VERSIONS
            ):
                raise ReplayCompatibilityError(
                    "replay schema mismatch: expected base schemas "
                    f"{SCHEMA_VERSIONS} with target in "
                    f"{sorted(SUPPORTED_TARGET_SCHEMA_VERSIONS)}, got {actual_versions}"
                )
            digest = hashlib.sha256()
            records = []
            for line in compressed:
                if not line.strip():
                    continue
                digest.update(line)
                record = ReplayRecord.from_dict(json.loads(line))
                record.validate_rules(engine)
                records.append(record)
    except ReplayCompatibilityError:
        raise
    except (OSError, EOFError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ReplayIntegrityError(f"invalid replay shard {path.name}: {error}") from error

    if len(records) != header.sample_count:
        raise ReplayIntegrityError("replay sample count does not match its header")
    if len({record.game_id for record in records}) != header.game_count:
        raise ReplayIntegrityError("replay game count does not match its header")
    if digest.hexdigest() != header.payload_sha256:
        raise ReplayIntegrityError("replay payload checksum mismatch")
    return ReplayShard(header, tuple(records))
