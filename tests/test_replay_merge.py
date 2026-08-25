from dataclasses import replace
from pathlib import Path

from test_replay_schema import scripted_game

from harbichess.replay.merge import merge_continuation_replay
from harbichess.replay.schema import records_from_game
from harbichess.replay.shard import ShardMetadata, read_shard, write_shard_atomic
from harbichess.replay.split import ReplaySplit


def _shard(
    path: Path,
    *,
    run_id: str,
    generation: int,
    created_at: str,
    records,
):
    write_shard_atomic(
        path,
        records,
        ShardMetadata(
            run_id=run_id,
            generation=generation,
            source_checkpoint="candidate",
            source_commit="a" * 40,
            created_at=created_at,
            split=ReplaySplit.TRAIN,
        ),
    )
    return read_shard(path)


def test_continuation_merge_keeps_latest_target_and_weights_recency(
    tmp_path: Path,
) -> None:
    _, game = scripted_game()
    all_records = records_from_game(game, run_id="old")
    base = all_records[:2]
    latest = tuple(
        replace(record, game_id=f"new-{index}", game_index=20 + index)
        for index, record in enumerate(base)
    )
    extra = replace(
        all_records[2],
        game_id="new-extra",
        game_index=30,
    )
    old_path = tmp_path / "old.jsonl.gz"
    new_path = tmp_path / "new.jsonl.gz"
    old = _shard(
        old_path,
        run_id="old",
        generation=1,
        created_at="2026-01-01T00:00:00+00:00",
        records=base,
    )
    new = _shard(
        new_path,
        run_id="new",
        generation=2,
        created_at="2026-02-01T00:00:00+00:00",
        records=(*latest, extra),
    )

    merged = merge_continuation_replay(
        ((new_path, new), (old_path, old)),
        recency_decay=0.5,
    )

    assert len(merged.records) == 3
    assert merged.duplicates_removed == 2
    assert {record.game_id for record in merged.records} == {
        "new-0",
        "new-1",
        "new-extra",
    }
    assert [source.recency_weight for source in merged.sources] == [0.5, 1.0]
    assert set(merged.game_weights.values()) == {1.0}
