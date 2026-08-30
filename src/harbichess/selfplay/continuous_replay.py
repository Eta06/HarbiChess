"""Generate small latest-network Full Gumbel replay generations."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from harbichess.backends.mlx_backend import MLXPolicyValueBackend
from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessState
from harbichess.replay.diversity import measure_diversity
from harbichess.replay.schema import ReplayRecord, records_from_game
from harbichess.search.batching import SharedBatchEvaluator
from harbichess.search.evaluator import NeuralPositionEvaluator
from harbichess.search.full_gumbel import FullGumbelConfig, FullGumbelMCTS
from harbichess.selfplay.game import (
    SelfPlayConfig,
    SelfPlayGame,
    play_parallel_games,
)


@dataclass(frozen=True, slots=True)
class ContinuousReplayConfig:
    games: int = 12
    workers: int = 12
    simulations: int = 64
    max_considered_actions: int = 16
    gumbel_scale: float = 1.0
    exploration_plies: int = 30
    max_plies: int = 96
    fixed_inference_batch_size: int = 12
    inference_wait_seconds: float = 0.00025

    def __post_init__(self) -> None:
        if (
            min(
                self.games,
                self.workers,
                self.simulations,
                self.max_considered_actions,
                self.max_plies,
                self.fixed_inference_batch_size,
            )
            <= 0
            or self.gumbel_scale <= 0
            or self.exploration_plies < 0
            or self.inference_wait_seconds < 0
        ):
            raise ValueError("continuous replay configuration is invalid")


def generate_continuous_replay(
    network,
    *,
    run_id: str,
    run_seed: int,
    config: ContinuousReplayConfig | None = None,
    on_game_complete: Callable[[SelfPlayGame], None] | None = None,
    initial_states: Sequence[ChessState] | None = None,
) -> tuple[tuple[SelfPlayGame, ...], tuple[ReplayRecord, ...], dict[str, object]]:
    """Play one versioned generation; max-ply outcomes remain unknown in replay."""

    settings = config or ContinuousReplayConfig()
    rules = PythonChessRules()
    batcher = SharedBatchEvaluator(
        MLXPolicyValueBackend(
            network,
            fixed_batch_size=settings.fixed_inference_batch_size,
        ),
        max_batch_size=min(settings.workers, settings.fixed_inference_batch_size),
        max_wait_seconds=settings.inference_wait_seconds,
    )
    evaluator = NeuralPositionEvaluator(batcher, rules=rules)
    search = FullGumbelMCTS(
        evaluator,
        rules=rules,
        config=FullGumbelConfig(
            simulations=settings.simulations,
            max_considered_actions=settings.max_considered_actions,
            gumbel_scale=settings.gumbel_scale,
        ),
    )
    started = time.perf_counter()
    try:
        games = play_parallel_games(
            search,  # type: ignore[arg-type]
            rules,
            run_seed=run_seed,
            first_game_index=0,
            game_count=settings.games,
            max_workers=min(settings.workers, settings.games),
            config=SelfPlayConfig(
                exploration_plies=settings.exploration_plies,
                max_plies=settings.max_plies,
                search_root_noise=False,
            ),
            on_game_complete=on_game_complete,
            initial_states=initial_states,
            max_additional_plies=settings.max_plies if initial_states is not None else None,
        )
    finally:
        batcher.close()
    elapsed = time.perf_counter() - started
    records = tuple(
        record for game in games for record in records_from_game(game, run_id=run_id, rules=rules)
    )
    diversity = measure_diversity(games)
    known_games = sum(game.outcome.termination != "max_plies" for game in games)
    known_records = sum(record.outcome_value is not None for record in records)
    start_phases = Counter(
        "opening"
        if game.samples[0].state.ply < 20
        else "middlegame"
        if game.samples[0].state.ply < 80
        else "endgame"
        for game in games
        if game.samples
    )
    return (
        games,
        records,
        {
            "elapsed_seconds": elapsed,
            "games": len(games),
            "positions": len(records),
            "known_outcome_games": known_games,
            "known_outcome_records": known_records,
            "start_phases": tuple(sorted(start_phases.items())),
            "terminations": tuple(
                sorted(
                    {
                        termination: sum(game.outcome.termination == termination for game in games)
                        for termination in {game.outcome.termination for game in games}
                    }.items()
                )
            ),
            "diversity": asdict(diversity),
            "inference": {
                **asdict(batcher.statistics),
                "positions_per_second": batcher.statistics.positions / max(elapsed, 1e-9),
            },
        },
    )
