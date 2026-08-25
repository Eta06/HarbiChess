"""Generation-aware continuation replay merge and recency weighting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harbichess.replay.schema import ReplayRecord
from harbichess.replay.shard import ReplayShard


@dataclass(frozen=True, slots=True)
class ContinuationSource:
    path: str
    run_id: str
    generation: int
    created_at: str
    samples: int
    recency_weight: float


@dataclass(frozen=True, slots=True)
class ContinuationMerge:
    records: tuple[ReplayRecord, ...]
    game_weights: dict[str, float]
    sources: tuple[ContinuationSource, ...]
    duplicates_removed: int


def merge_continuation_replay(
    shards: tuple[tuple[Path, ReplayShard], ...],
    *,
    recency_decay: float,
) -> ContinuationMerge:
    if not 0.0 < recency_decay <= 1.0:
        raise ValueError("continuation recency decay must be in (0, 1]")
    ordered = sorted(
        shards,
        key=lambda item: (
            item[1].header.generation,
            item[1].header.created_at,
            item[1].header.run_id,
        ),
    )
    by_position: dict[tuple[str, tuple[str, ...], str], tuple[ReplayRecord, float]] = {}
    sources = []
    for rank, (path, shard) in enumerate(ordered):
        age = len(ordered) - rank - 1
        weight = recency_decay**age
        sources.append(
            ContinuationSource(
                path=str(path),
                run_id=shard.header.run_id,
                generation=shard.header.generation,
                created_at=shard.header.created_at,
                samples=len(shard.records),
                recency_weight=weight,
            )
        )
        for record in shard.records:
            key = (record.root_fen, record.moves, record.side_to_move.value)
            by_position[key] = (record, weight)
    merged = tuple(
        record for record, _ in sorted(by_position.values(), key=lambda item: item[0].game_id)
    )
    game_weights = {record.game_id: weight for record, weight in by_position.values()}
    return ContinuationMerge(
        records=merged,
        game_weights=game_weights,
        sources=tuple(sources),
        duplicates_removed=sum(len(shard.records) for _, shard in ordered) - len(merged),
    )
