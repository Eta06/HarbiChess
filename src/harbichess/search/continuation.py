"""Repetition-aware MCTS target transformation without sacrificing draw defence."""

from __future__ import annotations

import random
from dataclasses import dataclass

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState
from harbichess.search.mcts import MoveStatistics, SearchResult


@dataclass(frozen=True, slots=True)
class ContinuationDecision:
    policy_moves: tuple[MoveStatistics, ...]
    selected_move: ChessMove
    transformed: bool
    repeating_policy_mass: float
    defensive_repetition_preserved: bool


def transform_repetition_target(
    search: SearchResult,
    rules: PythonChessRules,
    state: ChessState,
    selected: ChessMove,
    *,
    temperature: float,
    value_tolerance: float,
    minimum_repeating_policy_mass: float,
    rng: random.Random,
) -> ContinuationDecision:
    """Redirect meaningful repeat mass only when a comparable continuation exists."""

    if not 0.0 <= minimum_repeating_policy_mass <= 1.0:
        raise ValueError("minimum repeating policy mass must be in [0, 1]")
    visited = tuple(item for item in search.moves if item.visits > 0)
    total_visits = sum(item.visits for item in visited)
    if total_visits <= 0:
        raise ValueError("continuation targets require visited MCTS moves")
    moves_to_check = tuple(
        item.move
        for item in visited
        if item.move == selected or item.visits / total_visits >= minimum_repeating_policy_mass
    )
    repeating_moves = rules.claimable_threefold_moves(
        state,
        moves_to_check,
    )
    repeating = tuple(item for item in visited if item.move in repeating_moves)
    if not repeating:
        return ContinuationDecision(visited, selected, False, 0.0, False)
    repeating_mass = sum(item.visits for item in repeating) / total_visits
    selected_repeats = any(item.move == selected for item in repeating)
    if not selected_repeats and repeating_mass < minimum_repeating_policy_mass:
        return ContinuationDecision(visited, selected, False, repeating_mass, False)

    best_repeat_value = max(item.mean_value for item in repeating)
    comparable = tuple(
        item
        for item in visited
        if item not in repeating and item.mean_value >= best_repeat_value - value_tolerance
    )
    if not comparable:
        return ContinuationDecision(visited, selected, False, repeating_mass, True)

    redirected = SearchResult(
        comparable,
        search.root_value,
        search.simulations,
        search.outcome,
        search.network_priors,
    )
    return ContinuationDecision(
        policy_moves=comparable,
        selected_move=redirected.select_move(temperature=temperature, rng=rng),
        transformed=True,
        repeating_policy_mass=repeating_mass,
        defensive_repetition_preserved=False,
    )
