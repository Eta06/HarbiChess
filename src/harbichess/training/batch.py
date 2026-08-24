"""Framework-neutral training batches built from validated replay records."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from harbichess.chess.actions import POLICY_SIZE
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.backend import EncodedPosition
from harbichess.replay.schema import ReplayRecord


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    positions: tuple[EncodedPosition, ...]
    policy_targets: tuple[tuple[float, ...], ...]
    wdl_targets: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.positions)
        if size == 0 or len(self.policy_targets) != size or len(self.wdl_targets) != size:
            raise ValueError("training batch components must have equal non-zero length")
        if any(len(target) != POLICY_SIZE for target in self.policy_targets):
            raise ValueError(f"policy targets must contain {POLICY_SIZE} actions")
        if any(
            any(not math.isfinite(value) or value < 0 for value in target)
            or not math.isclose(sum(target), 1.0, abs_tol=1e-6)
            for target in self.policy_targets
        ):
            raise ValueError("policy targets must be finite, non-negative, and normalized")
        if any(target not in (0, 1, 2) for target in self.wdl_targets):
            raise ValueError("WDL class targets must be win=0, draw=1, or loss=2")


class GameBalancedSampler:
    """Sample games uniformly before selecting positions within each game."""

    def __init__(self, records: tuple[ReplayRecord, ...], *, seed: int) -> None:
        if not records:
            raise ValueError("sampler requires replay records")
        by_game: dict[str, list[ReplayRecord]] = {}
        for record in records:
            by_game.setdefault(record.game_id, []).append(record)
        self._by_game = {game_id: tuple(samples) for game_id, samples in by_game.items()}
        self._game_ids = tuple(sorted(self._by_game))
        self._rng = random.Random(seed)

    def sample(self, batch_size: int) -> tuple[ReplayRecord, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size <= len(self._game_ids):
            game_ids = self._rng.sample(self._game_ids, batch_size)
        else:
            game_ids = self._rng.choices(self._game_ids, k=batch_size)
        return tuple(self._rng.choice(self._by_game[game_id]) for game_id in game_ids)

    @property
    def rng_state(self) -> object:
        return self._rng.getstate()

    def set_rng_state(self, state: object) -> None:
        self._rng.setstate(state)


def build_training_batch(
    records: tuple[ReplayRecord, ...],
    *,
    rules: PythonChessRules | None = None,
    encoder: BoardEncoder | None = None,
) -> TrainingBatch:
    if not records:
        raise ValueError("cannot build an empty training batch")
    engine = rules or PythonChessRules()
    board_encoder = encoder or BoardEncoder(engine)
    positions = []
    policies = []
    wdl_targets = []
    for record in records:
        record.validate_rules(engine)
        positions.append(board_encoder.encode(record.state))
        dense_policy = [0.0] * POLICY_SIZE
        for action, probability in record.policy:
            dense_policy[action] = probability
        policies.append(tuple(dense_policy))
        wdl_targets.append({1: 0, 0: 1, -1: 2}[record.outcome_value])
    return TrainingBatch(tuple(positions), tuple(policies), tuple(wdl_targets))
