import gzip
import hashlib
import json
from datetime import UTC, datetime

import pytest
from test_replay_schema import scripted_game

from harbichess.replay.schema import TARGET_SCHEMA_VERSION, records_from_game
from harbichess.replay.shard import (
    ReplayCompatibilityError,
    ReplayIntegrityError,
    ShardMetadata,
    read_shard,
    write_shard_atomic,
)
from harbichess.replay.split import ReplaySplit


def metadata() -> ShardMetadata:
    return ShardMetadata(
        run_id="pilot",
        generation=0,
        source_checkpoint="champion-0000",
        source_commit="a" * 40,
        created_at=datetime.now(UTC).isoformat(),
        split=ReplaySplit.TRAIN,
    )


def rewrite_gzip(path, mutate) -> None:
    lines = gzip.decompress(path.read_bytes()).splitlines()
    parsed = [json.loads(line) for line in lines]
    mutate(parsed)
    payload = (
        b"\n".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() for value in parsed
        )
        + b"\n"
    )
    path.write_bytes(gzip.compress(payload, mtime=0))


def test_replay_shard_round_trips_and_leaves_no_temporary_files(tmp_path) -> None:
    rules, game = scripted_game()
    records = records_from_game(game, run_id="pilot", rules=rules)
    path = tmp_path / "shard-00000.jsonl.gz"

    header = write_shard_atomic(path, records, metadata())
    restored = read_shard(path, rules=rules)

    assert restored.header == header
    assert restored.records == records
    assert list(tmp_path.iterdir()) == [path]


def test_replay_shard_rejects_checksum_tampering(tmp_path) -> None:
    _, game = scripted_game()
    path = tmp_path / "tampered.jsonl.gz"
    write_shard_atomic(path, records_from_game(game, run_id="pilot"), metadata())
    rewrite_gzip(path, lambda lines: lines[1].__setitem__("root_value", 0.25))

    with pytest.raises(ReplayIntegrityError, match="checksum"):
        read_shard(path)


def test_replay_shard_rejects_future_schema(tmp_path) -> None:
    _, game = scripted_game()
    path = tmp_path / "future.jsonl.gz"
    write_shard_atomic(path, records_from_game(game, run_id="pilot"), metadata())
    rewrite_gzip(path, lambda lines: lines[0].__setitem__("replay_schema", 999))

    with pytest.raises(ReplayCompatibilityError, match="schema mismatch"):
        read_shard(path)

    write_shard_atomic(path, records_from_game(game, run_id="pilot"), metadata())
    rewrite_gzip(path, lambda lines: lines[0].__setitem__("target_schema", 999))
    with pytest.raises(ReplayCompatibilityError, match="schema mismatch"):
        read_shard(path)


def test_replay_shard_reads_legacy_target_schema_three(tmp_path) -> None:
    _, game = scripted_game()
    records = records_from_game(game, run_id="legacy")
    path = tmp_path / "legacy.jsonl.gz"
    write_shard_atomic(path, records, metadata())

    def downgrade(lines) -> None:
        lines[0]["target_schema"] = TARGET_SCHEMA_VERSION - 1
        for record in lines[1:]:
            record.pop("continuation_evidence")
        payload = b"".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for record in lines[1:]
        )
        lines[0]["payload_sha256"] = hashlib.sha256(payload).hexdigest()

    rewrite_gzip(path, downgrade)
    restored = read_shard(path)

    assert restored.header.target_schema == 3
    assert restored.records == records
