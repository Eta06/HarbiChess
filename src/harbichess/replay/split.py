"""Leakage-safe deterministic replay splitting at whole-game granularity."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import StrEnum

from harbichess.selfplay.game import SelfPlayGame


class ReplaySplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


def split_for_game(game_id: str, *, validation_fraction: float = 0.05) -> ReplaySplit:
    if not game_id or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("game_id is required and validation_fraction must be in [0, 1)")
    digest = hashlib.blake2b(game_id.encode(), digest_size=8).digest()
    fraction = int.from_bytes(digest) / 2**64
    return ReplaySplit.VALIDATION if fraction < validation_fraction else ReplaySplit.TRAIN


def partition_games(
    games: Iterable[SelfPlayGame],
    *,
    run_id: str,
    validation_fraction: float = 0.05,
) -> dict[ReplaySplit, tuple[SelfPlayGame, ...]]:
    partitions: dict[ReplaySplit, list[SelfPlayGame]] = {
        ReplaySplit.TRAIN: [],
        ReplaySplit.VALIDATION: [],
    }
    for game in games:
        game_id = f"{run_id}-{game.game_index:012d}"
        partitions[split_for_game(game_id, validation_fraction=validation_fraction)].append(game)
    return {split: tuple(split_games) for split, split_games in partitions.items()}
