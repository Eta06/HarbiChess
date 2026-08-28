"""Deterministic fixed-budget sequential halving at the search root."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from harbichess.core.state import ChessMove, ChessState
from harbichess.search.mcts import MCTS


@dataclass(frozen=True, slots=True)
class SequentialHalvingResult:
    selected_action: ChessMove
    considered_actions: tuple[ChessMove, ...]
    action_values: tuple[tuple[ChessMove, float], ...]
    action_slots: tuple[tuple[ChessMove, int], ...]
    rounds: int
    evaluation_slots: int


def _continuation_value(
    mcts: MCTS,
    state: ChessState,
    move: ChessMove,
    *,
    slots: int,
    rng: random.Random,
) -> float:
    if slots <= 0:
        raise ValueError("continuation evaluation slots must be positive")
    child = mcts.rules.apply(state, move)
    outcome = mcts.rules.outcome(child, claim_draw=mcts.config.claim_draw)
    if outcome is not None:
        child_side = mcts.rules.view(child).side_to_move
        return -float(outcome.value_for(child_side))
    if slots == 1:
        return -mcts.evaluator.evaluate(child).value
    result = MCTS(
        mcts.evaluator,
        rules=mcts.rules,
        config=replace(
            mcts.config,
            simulations=slots - 1,
            dirichlet_fraction=0.0,
        ),
    ).search(child, rng=rng, add_root_noise=False)
    return -result.root_value


def deterministic_sequential_halving(
    mcts: MCTS,
    state: ChessState,
    *,
    budget: int,
    rng: random.Random,
    maximum_considered_actions: int = 16,
) -> SequentialHalvingResult:
    """Allocate an exact number of search slots through root action elimination."""

    if budget <= 2 or maximum_considered_actions <= 1:
        raise ValueError("sequential halving requires budget > 2 and at least two actions")
    root = mcts.evaluator.evaluate(state)
    candidates = tuple(
        move
        for move, _ in sorted(root.priors, key=lambda item: (-item[1], item[0].uci))[
            :maximum_considered_actions
        ]
    )
    if not candidates:
        raise ValueError("sequential halving requires a non-terminal position")
    priors = dict(root.priors)
    used_slots = 1
    active = candidates
    rounds = max(1, math.ceil(math.log2(len(active))))
    weighted_values = {move: 0.0 for move in candidates}
    allocated_slots = {move: 0 for move in candidates}

    for round_index in range(rounds):
        rounds_left = rounds - round_index
        remaining = budget - used_slots
        per_action = max(1, remaining // (len(active) * rounds_left))
        if per_action * len(active) > remaining:
            raise ValueError("budget is too small for sequential halving schedule")
        for move in active:
            value = _continuation_value(mcts, state, move, slots=per_action, rng=rng)
            weighted_values[move] += value * per_action
            allocated_slots[move] += per_action
        used_slots += per_action * len(active)
        ranked = sorted(
            active,
            key=lambda move: (
                -weighted_values[move] / allocated_slots[move],
                -priors[move],
                move.uci,
            ),
        )
        active = tuple(ranked[: max(1, math.ceil(len(ranked) / 2))])

    winner = active[0]
    remaining = budget - used_slots
    if remaining:
        value = _continuation_value(mcts, state, winner, slots=remaining, rng=rng)
        weighted_values[winner] += value * remaining
        allocated_slots[winner] += remaining
        used_slots += remaining
    values = tuple(
        sorted(
            (
                (move, weighted_values[move] / allocated_slots[move])
                for move in candidates
                if allocated_slots[move]
            ),
            key=lambda item: item[0].uci,
        )
    )
    return SequentialHalvingResult(
        selected_action=winner,
        considered_actions=candidates,
        action_values=values,
        action_slots=tuple(sorted(allocated_slots.items(), key=lambda item: item[0].uci)),
        rounds=rounds,
        evaluation_slots=used_slots,
    )
