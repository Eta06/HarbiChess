"""Complete self-play game generation from PUCT visit distributions."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import (
    ChessMove,
    ChessState,
    GameOutcome,
    Side,
    TerminalResult,
)
from harbichess.search.continuation import transform_repetition_target
from harbichess.search.mcts import MCTS
from harbichess.search.root_halving import RootHalvingConfig, sequential_halving_root
from harbichess.search.value_policy import (
    ValueImprovedPolicyConfig,
    value_improved_policy,
)


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    exploration_plies: int = 30
    temperature: float = 1.0
    max_plies: int = 512
    repetition_value_tolerance: float = 0.05
    minimum_repeating_policy_mass: float = 0.10
    repetition_target_transform: bool = False
    value_policy_temperature: float | None = None
    value_policy_prior_visits: float = 8.0
    maximum_value_logit_adjustment: float = 1.25
    root_halving_config: RootHalvingConfig | None = None

    def __post_init__(self) -> None:
        if (
            self.exploration_plies < 0
            or self.temperature < 0
            or self.max_plies <= 0
            or not 0.0 <= self.repetition_value_tolerance <= 2.0
            or not 0.0 <= self.minimum_repeating_policy_mass <= 1.0
            or (
                self.value_policy_temperature is not None
                and self.value_policy_temperature <= 0
            )
            or self.value_policy_prior_visits < 0
            or self.maximum_value_logit_adjustment < 0
            or (
                self.root_halving_config is not None
                and self.value_policy_temperature is not None
            )
        ):
            raise ValueError("self-play temperatures and ply limits must be non-negative")


@dataclass(frozen=True, slots=True)
class SelfPlaySample:
    state: ChessState
    side_to_move: Side
    visit_policy: tuple[tuple[ChessMove, float], ...]
    selected_move: ChessMove
    root_value: float
    outcome_value: int | None
    repetition_redirected: bool = False
    root_search_adjusted: bool = False
    root_search_first_margin: float | None = None
    root_search_final_margin: float | None = None


@dataclass(frozen=True, slots=True)
class SelfPlayGame:
    game_index: int
    seed: int
    final_state: ChessState
    outcome: GameOutcome
    samples: tuple[SelfPlaySample, ...]


def derive_game_seed(run_seed: int, game_index: int) -> int:
    if game_index < 0:
        raise ValueError("game_index must be non-negative")
    payload = f"{run_seed}:{game_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest())


def play_game(
    mcts: MCTS,
    rules: PythonChessRules,
    initial_state: ChessState,
    *,
    game_index: int,
    seed: int,
    config: SelfPlayConfig | None = None,
) -> SelfPlayGame:
    settings = config or SelfPlayConfig()
    rng = random.Random(seed)
    state = initial_state
    pending: list[
        tuple[
            ChessState,
            Side,
            tuple[tuple[ChessMove, float], ...],
            ChessMove,
            float,
            bool,
            bool,
            float | None,
            float | None,
        ]
    ] = []

    while True:
        outcome = rules.outcome(state, claim_draw=True)
        if outcome is not None:
            break
        if state.ply >= settings.max_plies:
            outcome = GameOutcome(TerminalResult.DRAW, "max_plies")
            break
        search = mcts.search(state, rng=rng, add_root_noise=True)
        root_halving = (
            sequential_halving_root(
                mcts,
                state,
                search,
                rng=rng,
                config=settings.root_halving_config,
            )
            if settings.root_halving_config is not None
            else None
        )
        if root_halving is not None:
            search = root_halving.search
        root_evidence = root_halving.evidence if root_halving is not None else None
        side_to_move = rules.view(state).side_to_move
        temperature = settings.temperature if state.ply < settings.exploration_plies else 0.0
        selected = search.select_move(temperature=temperature, rng=rng)
        if settings.repetition_target_transform:
            continuation = transform_repetition_target(
                search,
                rules,
                state,
                selected,
                temperature=temperature,
                value_tolerance=settings.repetition_value_tolerance,
                minimum_repeating_policy_mass=settings.minimum_repeating_policy_mass,
                rng=rng,
            )
            policy_moves = continuation.policy_moves
            selected = continuation.selected_move
            repetition_redirected = continuation.transformed
        else:
            policy_moves = search.moves
            repetition_redirected = False
        if settings.value_policy_temperature is None:
            total_visits = sum(statistics.visits for statistics in policy_moves)
            if total_visits <= 0:
                raise RuntimeError("non-terminal search returned no visited moves")
            policy = tuple(
                (statistics.move, statistics.visits / total_visits)
                for statistics in policy_moves
                if statistics.visits > 0
            )
        else:
            policy = value_improved_policy(
                policy_moves,
                search.root_value,
                config=ValueImprovedPolicyConfig(
                    advantage_temperature=settings.value_policy_temperature,
                    prior_visits=settings.value_policy_prior_visits,
                    maximum_logit_adjustment=(
                        settings.maximum_value_logit_adjustment
                    ),
                ),
            )
        pending.append(
            (
                state,
                side_to_move,
                policy,
                selected,
                search.root_value,
                repetition_redirected,
                bool(root_evidence and root_evidence.adjusted),
                root_evidence.first_round_margin if root_evidence else None,
                root_evidence.final_round_margin if root_evidence else None,
            )
        )
        state = rules.apply(state, selected)

    samples = tuple(
        SelfPlaySample(
            state=sample_state,
            side_to_move=side,
            visit_policy=policy,
            selected_move=selected_move,
            root_value=root_value,
            outcome_value=(
                None if outcome.termination == "max_plies" else outcome.value_for(side)
            ),
            repetition_redirected=repetition_redirected,
            root_search_adjusted=root_search_adjusted,
            root_search_first_margin=root_search_first_margin,
            root_search_final_margin=root_search_final_margin,
        )
        for (
            sample_state,
            side,
            policy,
            selected_move,
            root_value,
            repetition_redirected,
            root_search_adjusted,
            root_search_first_margin,
            root_search_final_margin,
        ) in pending
    )
    return SelfPlayGame(game_index, seed, state, outcome, samples)


def play_parallel_games(
    mcts: MCTS,
    rules: PythonChessRules,
    *,
    run_seed: int,
    first_game_index: int,
    game_count: int,
    max_workers: int,
    config: SelfPlayConfig | None = None,
    on_game_complete: Callable[[SelfPlayGame], None] | None = None,
) -> tuple[SelfPlayGame, ...]:
    if game_count <= 0 or max_workers <= 0:
        raise ValueError("game_count and max_workers must be positive")
    initial_state = rules.initial_state()

    def play(game_index: int) -> SelfPlayGame:
        game = play_game(
            mcts,
            rules,
            initial_state,
            game_index=game_index,
            seed=derive_game_seed(run_seed, game_index),
            config=config,
        )
        if on_game_complete is not None:
            on_game_complete(game)
        return game

    indices = range(first_game_index, first_game_index + game_count)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="harbichess-game") as pool:
        return tuple(pool.map(play, indices))
