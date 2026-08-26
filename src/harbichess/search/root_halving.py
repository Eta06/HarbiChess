"""Fixed-budget confidence-gated sequential halving at the search root."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from harbichess.core.state import ChessMove, ChessState
from harbichess.search.mcts import MCTS, MoveStatistics, SearchResult


@dataclass(frozen=True, slots=True)
class RootHalvingConfig:
    top_actions: int = 4
    finalists: int = 2
    first_round_simulations: int = 3
    final_round_simulations: int = 7
    minimum_consensus_margin: float = 0.05
    transfer_fraction: float = 0.35
    policy_scale: int = 10_000

    def __post_init__(self) -> None:
        if (
            self.top_actions < 2
            or not 1 < self.finalists < self.top_actions
            or self.first_round_simulations <= 0
            or self.final_round_simulations <= self.first_round_simulations
            or not 0.0 <= self.minimum_consensus_margin <= 2.0
            or not 0.0 < self.transfer_fraction < 1.0
            or self.policy_scale <= 0
        ):
            raise ValueError("root-halving configuration is invalid")

    @property
    def forced_evaluations(self) -> int:
        return self.top_actions * (self.first_round_simulations + 1) + self.finalists * (
            self.final_round_simulations + 1
        )


@dataclass(frozen=True, slots=True)
class RootHalvingEvidence:
    candidates: tuple[ChessMove, ...]
    finalists: tuple[ChessMove, ...]
    winner: ChessMove
    visit_leader: ChessMove
    first_round_margin: float
    final_round_margin: float
    original_winner_mass: float
    adjusted_winner_mass: float
    adjusted: bool


@dataclass(frozen=True, slots=True)
class RootHalvingResult:
    search: SearchResult
    evidence: RootHalvingEvidence | None


def _continuation_value(
    mcts: MCTS,
    state: ChessState,
    move: ChessMove,
    *,
    simulations: int,
    rng: random.Random,
) -> float:
    child = mcts.rules.apply(state, move)
    continuation = MCTS(
        mcts.evaluator,
        rules=mcts.rules,
        config=replace(
            mcts.config,
            simulations=simulations,
            dirichlet_fraction=0.0,
        ),
    ).search(child, rng=rng, add_root_noise=False)
    return -continuation.root_value


def sequential_halving_root(
    mcts: MCTS,
    state: ChessState,
    initial: SearchResult,
    *,
    rng: random.Random,
    config: RootHalvingConfig | None = None,
) -> RootHalvingResult:
    """Spend a fixed continuation budget and adjust only consensus winners."""

    settings = config or RootHalvingConfig()
    visited = tuple(move for move in initial.moves if move.visits > 0)
    if len(visited) < settings.top_actions:
        return RootHalvingResult(initial, None)
    candidates = visited[: settings.top_actions]
    first_values = {
        move.move: _continuation_value(
            mcts,
            state,
            move.move,
            simulations=settings.first_round_simulations,
            rng=rng,
        )
        for move in candidates
    }
    first_ranking = sorted(
        candidates,
        key=lambda move: (-first_values[move.move], -move.visits, move.move.uci),
    )
    finalists = tuple(first_ranking[: settings.finalists])
    final_values = {
        move.move: _continuation_value(
            mcts,
            state,
            move.move,
            simulations=settings.final_round_simulations,
            rng=rng,
        )
        for move in finalists
    }
    final_ranking = sorted(
        finalists,
        key=lambda move: (-final_values[move.move], -move.visits, move.move.uci),
    )
    winner, runner_up = final_ranking[:2]
    first_runner = next(move for move in first_ranking if move.move != winner.move)
    first_margin = first_values[winner.move] - first_values[first_runner.move]
    final_margin = final_values[winner.move] - final_values[runner_up.move]
    adjusted = (
        first_ranking[0].move == winner.move
        and first_margin >= settings.minimum_consensus_margin
        and final_margin >= settings.minimum_consensus_margin
    )

    total_visits = sum(move.visits for move in visited)
    base_policy = {move.move: move.visits / total_visits for move in visited}
    original_winner_mass = base_policy[winner.move]
    adjusted_policy = dict(base_policy)
    if adjusted:
        transfer = settings.transfer_fraction * sum(
            base_policy[move.move] for move in candidates if move.move != winner.move
        )
        for move in candidates:
            if move.move != winner.move:
                adjusted_policy[move.move] *= 1.0 - settings.transfer_fraction
        adjusted_policy[winner.move] += transfer

    refined_values = dict(first_values)
    refined_values.update(final_values)
    refined_moves = tuple(
        MoveStatistics(
            move.move,
            (
                max(1, round(adjusted_policy[move.move] * settings.policy_scale))
                if adjusted and move.visits > 0
                else move.visits
            ),
            move.prior,
            refined_values.get(move.move, move.mean_value),
        )
        for move in initial.moves
    )
    search = SearchResult(
        refined_moves,
        initial.root_value,
        initial.simulations + settings.forced_evaluations,
        initial.outcome,
    )
    return RootHalvingResult(
        search,
        RootHalvingEvidence(
            candidates=tuple(move.move for move in candidates),
            finalists=tuple(move.move for move in finalists),
            winner=winner.move,
            visit_leader=visited[0].move,
            first_round_margin=first_margin,
            final_round_margin=final_margin,
            original_winner_mass=original_winner_mass,
            adjusted_winner_mass=adjusted_policy[winner.move],
            adjusted=adjusted,
        ),
    )
