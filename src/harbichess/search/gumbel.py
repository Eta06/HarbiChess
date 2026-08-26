"""Gumbel Top-k root action sampling with fixed-budget sequential halving."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from harbichess.core.state import ChessMove, ChessState
from harbichess.search.mcts import MCTS


@dataclass(frozen=True, slots=True)
class GumbelSearchConfig:
    simulations: int
    top_actions: int = 16
    value_scale: float = 1.0
    policy_scale: int = 10_000

    def __post_init__(self) -> None:
        if (
            self.simulations <= 0
            or self.top_actions <= 1
            or not math.isfinite(self.value_scale)
            or self.value_scale <= 0
            or self.policy_scale <= 0
        ):
            raise ValueError("Gumbel search configuration is invalid")


@dataclass(frozen=True, slots=True)
class GumbelSearchResult:
    policy: tuple[tuple[ChessMove, float], ...]
    selected_move: ChessMove
    action_values: tuple[tuple[ChessMove, float], ...]
    sampled_actions: tuple[ChessMove, ...]
    simulations: int


def _gumbel(rng: random.Random) -> float:
    uniform = min(1.0 - 1e-12, max(1e-12, rng.random()))
    return -math.log(-math.log(uniform))


def _softmax(items: tuple[tuple[ChessMove, float], ...]) -> tuple[tuple[ChessMove, float], ...]:
    maximum = max(score for _, score in items)
    weights = tuple((move, math.exp(score - maximum)) for move, score in items)
    total = sum(weight for _, weight in weights)
    return tuple((move, weight / total) for move, weight in weights)


def _continuation_value(
    mcts: MCTS,
    state: ChessState,
    move: ChessMove,
    *,
    simulations: int,
    rng: random.Random,
) -> float:
    child = mcts.rules.apply(state, move)
    result = MCTS(
        mcts.evaluator,
        rules=mcts.rules,
        config=replace(
            mcts.config,
            simulations=simulations,
            dirichlet_fraction=0.0,
        ),
    ).search(child, rng=rng, add_root_noise=False)
    return -result.root_value


def gumbel_sequential_halving(
    mcts: MCTS,
    state: ChessState,
    *,
    rng: random.Random,
    config: GumbelSearchConfig,
) -> GumbelSearchResult:
    """Apply the Gumbel AlphaZero root operator without replacement.

    Actions are sampled by Gumbel Top-k, evaluated through independent child
    searches, and repeatedly halved under one fixed continuation budget. The
    training policy uses completed action values and excludes the sampled Gumbel
    perturbations, matching their roles in the policy-improvement operator.
    """

    root = mcts.evaluator.evaluate(state)
    priors = dict(root.priors)
    if len(priors) < 2:
        raise ValueError("Gumbel search requires at least two legal actions")
    candidate_count = min(config.top_actions, config.simulations, len(priors))
    perturbations = {move: _gumbel(rng) for move in priors}
    logits = {move: math.log(max(prior, 1e-12)) for move, prior in priors.items()}
    sampled = tuple(
        sorted(
            priors,
            key=lambda move: (-(logits[move] + perturbations[move]), move.uci),
        )[:candidate_count]
    )
    active = sampled
    value_sums = {move: 0.0 for move in sampled}
    value_visits = {move: 0 for move in sampled}
    remaining = config.simulations

    while len(active) > 1 and remaining >= len(active):
        rounds_left = max(1, math.ceil(math.log2(len(active))))
        per_action = max(1, remaining // (len(active) * rounds_left))
        per_action = min(per_action, remaining // len(active))
        for move in active:
            value = _continuation_value(
                mcts,
                state,
                move,
                simulations=per_action,
                rng=rng,
            )
            value_sums[move] += value * per_action
            value_visits[move] += per_action
            remaining -= per_action
        active = tuple(
            sorted(
                active,
                key=lambda move: (
                    -(
                        logits[move]
                        + perturbations[move]
                        + config.value_scale * value_sums[move] / value_visits[move]
                    ),
                    move.uci,
                ),
            )[: max(1, math.ceil(len(active) / 2))]
        )

    winner = active[0]
    if remaining > 0:
        value = _continuation_value(
            mcts,
            state,
            winner,
            simulations=remaining,
            rng=rng,
        )
        value_sums[winner] += value * remaining
        value_visits[winner] += remaining
        remaining = 0

    completed_values = {
        move: (
            value_sums[move] / value_visits[move]
            if move in value_visits and value_visits[move] > 0
            else root.value
        )
        for move in priors
    }
    policy = _softmax(
        tuple(
            (
                move,
                logits[move] + config.value_scale * (completed_values[move] - root.value),
            )
            for move in priors
        )
    )
    return GumbelSearchResult(
        policy=policy,
        selected_move=winner,
        action_values=tuple(sorted(completed_values.items(), key=lambda item: item[0].uci)),
        sampled_actions=sampled,
        simulations=config.simulations - remaining,
    )
