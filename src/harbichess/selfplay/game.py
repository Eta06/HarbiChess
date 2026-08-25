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
from harbichess.search.mcts import MCTS, MoveStatistics, SearchResult


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    exploration_plies: int = 30
    temperature: float = 1.0
    max_plies: int = 512
    repetition_value_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if (
            self.exploration_plies < 0
            or self.temperature < 0
            or self.max_plies <= 0
            or not 0.0 <= self.repetition_value_tolerance <= 2.0
        ):
            raise ValueError("self-play temperatures and ply limits must be non-negative")


@dataclass(frozen=True, slots=True)
class SelfPlaySample:
    state: ChessState
    side_to_move: Side
    visit_policy: tuple[tuple[ChessMove, float], ...]
    selected_move: ChessMove
    root_value: float
    outcome_value: int
    repetition_redirected: bool = False


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


def _immediate_threefold(
    rules: PythonChessRules,
    state: ChessState,
    move: ChessMove,
) -> bool:
    outcome = rules.outcome(rules.apply(state, move), claim_draw=True)
    return outcome is not None and outcome.termination == "threefold_repetition"


def _redirect_repetition(
    search: SearchResult,
    rules: PythonChessRules,
    state: ChessState,
    selected: ChessMove,
    *,
    temperature: float,
    tolerance: float,
    rng: random.Random,
) -> tuple[tuple[MoveStatistics, ...], ChessMove, bool]:
    if not _immediate_threefold(rules, state, selected):
        return search.moves, selected, False
    selected_statistics = next(item for item in search.moves if item.move == selected)
    alternatives = tuple(
        item
        for item in search.moves
        if item.visits > 0
        and not _immediate_threefold(rules, state, item.move)
        and item.mean_value >= selected_statistics.mean_value - tolerance
    )
    if not alternatives:
        return search.moves, selected, False
    redirected_search = SearchResult(
        alternatives,
        search.root_value,
        search.simulations,
        search.outcome,
    )
    return (
        alternatives,
        redirected_search.select_move(temperature=temperature, rng=rng),
        True,
    )


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
        side_to_move = rules.view(state).side_to_move
        temperature = settings.temperature if state.ply < settings.exploration_plies else 0.0
        selected = search.select_move(temperature=temperature, rng=rng)
        policy_moves, selected, repetition_redirected = _redirect_repetition(
            search,
            rules,
            state,
            selected,
            temperature=temperature,
            tolerance=settings.repetition_value_tolerance,
            rng=rng,
        )
        total_visits = sum(statistics.visits for statistics in policy_moves)
        if total_visits <= 0:
            raise RuntimeError("non-terminal search returned no visited moves")
        policy = tuple(
            (statistics.move, statistics.visits / total_visits)
            for statistics in policy_moves
            if statistics.visits > 0
        )
        pending.append(
            (
                state,
                side_to_move,
                policy,
                selected,
                search.root_value,
                repetition_redirected,
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
            outcome_value=outcome.value_for(side),
            repetition_redirected=repetition_redirected,
        )
        for (
            sample_state,
            side,
            policy,
            selected_move,
            root_value,
            repetition_redirected,
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
