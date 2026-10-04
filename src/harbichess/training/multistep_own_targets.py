"""Experimental retrospective own-experience WDL targets; no enabled learner.

A block contains H complete actor steps recorded under ONE unchanged current
policy and EMA. Every transition uses its own remaining same-game suffix, up to
the block boundary. Actual terminals supersede bootstrap; actor caps close an
UNKNOWN game and require EMA at that game's final post-state. Reset openings
never supply targets to the previous game. H=1 has the existing one-ply scalar
semantics. There is no discount or external teacher/search label.

This pure helper cannot establish that model snapshots stayed frozen, that pi
and mu were computed over identical legal support, or that a checkpoint is
complete. Those remain collector/learner responsibilities. In particular the
existing one-ply native-v1 checkpoint counters MUST NOT be reused for H>1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from harbichess.chess.rules import PythonChessRules
from harbichess.selfplay.online_actor import ActorTransition
from harbichess.training.online_targets import (
    WDL,
    clipped_importance,
    expected_score,
    flip_wdl,
    validate_wdl,
)

MULTISTEP_OWN_TARGET_SCHEMA = "fixed-block-own-suffix-stm-wdl-v1"


@dataclass(frozen=True, slots=True)
class MultistepOwnTarget:
    schema: str
    transition: ActorTransition
    leaf_index: int
    credit_plies: int
    block_horizon: int
    claim_draw: bool
    target_source: str
    leaf_rollout_cutoff: bool
    terminal_result: str | None
    terminal_termination: str | None
    mover_wdl: WDL
    advantage: float
    importance: float


def _segments(rules, transitions, *, horizon, claim_draw):
    if type(horizon) is not int or horizon <= 0 or type(claim_draw) is not bool:
        raise ValueError("positive integer horizon and explicit boolean claim_draw required")
    if not transitions or len(transitions) % horizon:
        raise ValueError("block must contain exactly horizon complete actor steps")
    games = len(transitions) // horizon
    previous = {}
    segments = {}
    game_owners = {}
    # Require the OnlineActors.step step-major shape, including every slot.
    for index, row in enumerate(transitions):
        if (
            type(row.slot) is not int
            or row.slot != index % games
            or type(row.game_index) is not int
            or row.game_index < 0
            or type(row.rollout_cutoff) is not bool
            or not isinstance(row.source_id, str)
            or not row.source_id.strip()
        ):
            raise ValueError("invalid complete actor-step order or transition metadata")
        if rules.outcome(row.pre, claim_draw=claim_draw) is not None:
            raise ValueError("cannot learn an action after a terminal position")
        if rules.apply(row.pre, row.action) != row.post:
            raise ValueError("recorded post-state differs from full legal history")
        outcome = rules.outcome(row.post, claim_draw=claim_draw)
        if (
            row.terminal_result != (outcome.result.value if outcome else None)
            or row.terminal_termination != (outcome.termination if outcome else None)
        ):
            raise ValueError("recorded terminal metadata differs from rules outcome")
        key = row.slot, row.game_index
        if game_owners.setdefault(row.game_index, row.slot) != row.slot:
            raise ValueError("game ID shared by multiple actor slots")
        if row.slot in previous:
            prior = transitions[previous[row.slot]]
            closed = prior.terminal_result is not None or prior.rollout_cutoff
            if row.game_index == prior.game_index:
                if closed or prior.post != row.pre or prior.source_id != row.source_id:
                    raise ValueError("same-game history is discontinuous or follows closure")
            elif not closed or row.game_index <= prior.game_index or key in segments:
                raise ValueError("game reset without closure or reused game ID")
        previous[row.slot] = index
        segments.setdefault(key, []).append(index)
    return tuple(tuple(indices) for indices in segments.values())


def required_multistep_bootstrap_indices(
    rules: PythonChessRules,
    transitions: tuple[ActorTransition, ...],
    *,
    horizon: int,
    claim_draw: bool,
) -> tuple[int, ...]:
    """Return only nonterminal same-game leaf indices to query with frozen EMA.

    Query transitions[index].post, never the actor's reset state. Includes
    UNKNOWN rollout caps; excludes real terminals even at simultaneous caps.
    Validation is read-only and consumes neither actor nor global RNG.
    """
    segments = _segments(rules, transitions, horizon=horizon, claim_draw=claim_draw)
    return tuple(sorted(indices[-1] for indices in segments
                        if transitions[indices[-1]].terminal_result is None))


def build_multistep_own_targets(
    rules: PythonChessRules,
    transitions: tuple[ActorTransition, ...],
    *,
    online_pre_wdl: tuple[WDL, ...],
    ema_leaf_wdl: Mapping[int, WDL],
    horizon: int,
    importance_maximum: float,
    claim_draw: bool,
) -> tuple[MultistepOwnTarget, ...]:
    """Freeze one target per recorded action in original step-major order.

    EMA leaves and online baselines must be detached W,D,L probabilities in
    their respective states' side-to-move perspective. Importance is the
    recorded action's clipped pi/mu, not a trajectory-ratio product. With
    temperature=1 and one frozen current policy, pi=mu. Other temperatures are
    an explicit biased multistep approximation: single-action importance does
    not correct the future behavior trajectory. No off-policy claim is made.
    """
    segments = _segments(rules, transitions, horizon=horizon, claim_draw=claim_draw)
    if len(online_pre_wdl) != len(transitions):
        raise ValueError("one frozen online baseline required per transition")
    current = tuple(validate_wdl(wdl) for wdl in online_pre_wdl)
    required = {indices[-1] for indices in segments
                if transitions[indices[-1]].terminal_result is None}
    if any(type(index) is not int for index in ema_leaf_wdl) or set(ema_leaf_wdl) != required:
        raise ValueError("EMA mapping must contain exactly the nonterminal same-game leaves")
    bootstrap = {index: validate_wdl(wdl) for index, wdl in ema_leaf_wdl.items()}
    targets = [None] * len(transitions)
    for indices in segments:
        leaf_index = indices[-1]
        leaf = transitions[leaf_index]
        outcome = rules.outcome(leaf.post, claim_draw=claim_draw)
        for offset, index in enumerate(indices):
            row = transitions[index]
            plies = len(indices) - offset
            if outcome is not None:
                value = outcome.value_for(rules.view(row.pre).side_to_move)
                wdl = (float(value == 1), float(value == 0), float(value == -1))
                source = "observed-terminal"
            else:
                wdl = flip_wdl(bootstrap[leaf_index]) if plies % 2 else bootstrap[leaf_index]
                source = "ema-bootstrap"
            targets[index] = MultistepOwnTarget(
                schema=MULTISTEP_OWN_TARGET_SCHEMA,
                transition=row,
                leaf_index=leaf_index,
                credit_plies=plies,
                block_horizon=horizon,
                claim_draw=claim_draw,
                target_source=source,
                leaf_rollout_cutoff=leaf.rollout_cutoff,
                terminal_result=leaf.terminal_result,
                terminal_termination=leaf.terminal_termination,
                mover_wdl=wdl,
                advantage=expected_score(wdl) - expected_score(current[index]),
                importance=clipped_importance(
                    row.policy_probability, row.behavior_probability, maximum=importance_maximum
                ),
            )
    return tuple(targets)
