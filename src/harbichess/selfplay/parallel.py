"""Parallel root searches with isolated deterministic game randomness."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from harbichess.core.state import ChessMove, ChessState
from harbichess.search.mcts import MCTS, SearchResult


@dataclass(frozen=True, slots=True)
class GameSearchResult:
    game_seed: int
    search: SearchResult
    selected_move: ChessMove | None


def run_parallel_searches(
    mcts: MCTS,
    states: list[ChessState],
    game_seeds: list[int],
    *,
    max_workers: int,
    temperature: float,
) -> tuple[GameSearchResult, ...]:
    """Search game roots concurrently without sharing their random streams."""
    if len(states) != len(game_seeds):
        raise ValueError("every state must have exactly one game seed")
    if len(set(game_seeds)) != len(game_seeds):
        raise ValueError("game seeds must be unique within a parallel wave")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    def search_one(item: tuple[ChessState, int]) -> GameSearchResult:
        state, seed = item
        rng = random.Random(seed)
        result = mcts.search(state, rng=rng, add_root_noise=True)
        selected = result.select_move(temperature=temperature, rng=rng) if result.moves else None
        return GameSearchResult(seed, result, selected)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="harbichess-game") as pool:
        return tuple(pool.map(search_one, zip(states, game_seeds, strict=True)))
