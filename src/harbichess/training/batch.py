"""Framework-neutral training batches built from validated replay records."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import InitVar, dataclass

from harbichess.chess.actions import POLICY_SIZE, legal_action_indices
from harbichess.chess.encoding import BoardEncoder
from harbichess.chess.rules import PythonChessRules
from harbichess.core.backend import EncodedPosition
from harbichess.replay.schema import ReplayRecord


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    positions: tuple[EncodedPosition, ...]
    policy_targets: tuple[tuple[float, ...], ...]
    legal_masks: tuple[tuple[bool, ...], ...]
    wdl_targets: tuple[int, ...]
    value_weights: tuple[float, ...]
    _validate: InitVar[bool] = True

    def __post_init__(self, _validate: bool) -> None:
        size = len(self.positions)
        if (
            size == 0
            or len(self.policy_targets) != size
            or len(self.legal_masks) != size
            or len(self.wdl_targets) != size
            or len(self.value_weights) != size
        ):
            raise ValueError("training batch components must have equal non-zero length")
        if not _validate:
            return
        if any(len(target) != POLICY_SIZE for target in self.policy_targets):
            raise ValueError(f"policy targets must contain {POLICY_SIZE} actions")
        if any(len(mask) != POLICY_SIZE or not any(mask) for mask in self.legal_masks):
            raise ValueError(f"legal masks must contain {POLICY_SIZE} actions and legal moves")
        if any(
            any(not math.isfinite(value) or value < 0 for value in target)
            or not math.isclose(sum(target), 1.0, abs_tol=1e-6)
            for target in self.policy_targets
        ):
            raise ValueError("policy targets must be finite, non-negative, and normalized")
        if any(target not in (0, 1, 2) for target in self.wdl_targets):
            raise ValueError("WDL class targets must be win=0, draw=1, or loss=2")
        if any(weight not in (0.0, 1.0) for weight in self.value_weights):
            raise ValueError("value weights must mask unknown targets with zero or one")

    def select(self, indices: tuple[int, ...]) -> TrainingBatch:
        """Select rows already validated by this immutable parent batch."""

        if not indices or any(index < 0 or index >= len(self.positions) for index in indices):
            raise IndexError("training batch indices must be non-empty and in range")
        return TrainingBatch(
            tuple(self.positions[index] for index in indices),
            tuple(self.policy_targets[index] for index in indices),
            tuple(self.legal_masks[index] for index in indices),
            tuple(self.wdl_targets[index] for index in indices),
            tuple(self.value_weights[index] for index in indices),
            _validate=False,
        )


class GameBalancedSampler:
    """Sample games uniformly before selecting positions within each game."""

    def __init__(
        self,
        records: tuple[ReplayRecord, ...],
        *,
        seed: int,
        continuation_fraction: float | None = None,
        continuation_game_weights: Mapping[str, float] | None = None,
    ) -> None:
        if not records:
            raise ValueError("sampler requires replay records")
        if continuation_fraction is not None and not 0.0 <= continuation_fraction <= 1.0:
            raise ValueError("continuation fraction must be in [0, 1]")
        if continuation_game_weights is not None and any(
            not math.isfinite(weight) or weight <= 0
            for weight in continuation_game_weights.values()
        ):
            raise ValueError("continuation game weights must be finite and positive")
        index_by_game: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            index_by_game.setdefault(record.game_id, []).append(index)
        self._records = records
        self._indices_by_game = {
            game_id: tuple(indices) for game_id, indices in index_by_game.items()
        }
        self._game_ids = tuple(sorted(self._indices_by_game))
        self._continuation_game_ids = tuple(
            game_id
            for game_id in self._game_ids
            if any(records[index].repetition_redirected for index in self._indices_by_game[game_id])
        )
        continuation_games = set(self._continuation_game_ids)
        self._standard_game_ids = tuple(
            game_id for game_id in self._game_ids if game_id not in continuation_games
        )
        self._continuation_fraction = continuation_fraction
        self._continuation_game_weights = continuation_game_weights
        self._rng = random.Random(seed)

    def sample(self, batch_size: int) -> tuple[ReplayRecord, ...]:
        return tuple(self._records[index] for index in self.sample_indices(batch_size))

    def sample_indices(self, batch_size: int) -> tuple[int, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._continuation_fraction is None or not self._continuation_game_ids:
            game_ids = self._sample_games(self._game_ids, batch_size)
        elif not self._standard_game_ids:
            game_ids = self._sample_continuation_games(batch_size)
        else:
            continuation_count = round(batch_size * self._continuation_fraction)
            standard_count = batch_size - continuation_count
            game_ids = [
                *self._sample_continuation_games(continuation_count),
                *self._sample_games(self._standard_game_ids, standard_count),
            ]
            self._rng.shuffle(game_ids)
        return tuple(self._rng.choice(self._indices_by_game[game_id]) for game_id in game_ids)

    def _sample_games(self, game_ids: tuple[str, ...], count: int) -> list[str]:
        if count <= 0:
            return []
        if count <= len(game_ids):
            return self._rng.sample(game_ids, count)
        return self._rng.choices(game_ids, k=count)

    def _sample_continuation_games(self, count: int) -> list[str]:
        if self._continuation_game_weights is None:
            return self._sample_games(self._continuation_game_ids, count)
        weights = [
            self._continuation_game_weights.get(game_id, 1.0)
            for game_id in self._continuation_game_ids
        ]
        return self._rng.choices(self._continuation_game_ids, weights=weights, k=count)

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
    legal_masks = []
    wdl_targets = []
    value_weights = []
    for record in records:
        board = engine.board(record.state)
        record.validate_board(board)
        positions.append(board_encoder.encode_state(record.state, board))
        dense_policy = [0.0] * POLICY_SIZE
        for action, probability in record.policy:
            dense_policy[action] = probability
        policies.append(tuple(dense_policy))
        legal = [False] * POLICY_SIZE
        for action in legal_action_indices(board):
            legal[action] = True
        legal_masks.append(tuple(legal))
        if record.outcome_value is None:
            wdl_targets.append(1)
            value_weights.append(0.0)
        else:
            wdl_targets.append({1: 0, 0: 1, -1: 2}[record.outcome_value])
            value_weights.append(1.0)
    return TrainingBatch(
        tuple(positions),
        tuple(policies),
        tuple(legal_masks),
        tuple(wdl_targets),
        tuple(value_weights),
    )
