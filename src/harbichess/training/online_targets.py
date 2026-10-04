"""Versioned one-ply self-play targets, independent of a numerical backend.

This is an experimental building block, not an enabled training loop. A cutoff
is not a terminal draw. Nonterminal targets bootstrap the next player's WDL;
real terminal targets use the mover's result. Callers must explicitly choose the
draw-claim convention and supply probabilities recorded before any update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from harbichess.chess.rules import PythonChessRules
from harbichess.core.state import ChessMove, ChessState

ONLINE_TARGET_SCHEMA = "one-ply-stm-wdl-v1"
WDL = tuple[float, float, float]


def validate_wdl(probabilities: WDL) -> WDL:
    """Validate an actual W,D,L distribution without silently normalizing it."""
    if len(probabilities) != 3:
        raise ValueError("WDL must have exactly three entries in W,D,L order")
    values = tuple(float(value) for value in probabilities)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("WDL probabilities must be finite and within [0,1]")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("WDL probabilities must sum to one")
    return values


def expected_score(probabilities: WDL) -> float:
    win, draw, _ = validate_wdl(probabilities)
    return win + 0.5 * draw


def flip_wdl(probabilities: WDL) -> WDL:
    win, draw, loss = validate_wdl(probabilities)
    return loss, draw, win


def clipped_importance(
    policy_probability: float, behavior_probability: float, *, maximum: float
) -> float:
    """Recorded pi(a|s)/mu(a|s), truncated at an explicit finite positive cap.

    Both distributions must already be normalized over the same legal actions.
    This scalar helper cannot establish that support/provenance condition.
    """
    if (
        not math.isfinite(policy_probability)
        or not 0 <= policy_probability <= 1
        or not math.isfinite(behavior_probability)
        or not 0 < behavior_probability <= 1
        or not math.isfinite(maximum)
        or maximum <= 0
    ):
        raise ValueError("invalid recorded action probability or importance cap")
    # Avoid an overflowing intermediate ratio for tiny proposal probabilities.
    if policy_probability == 0:
        return 0.0
    if policy_probability >= maximum * behavior_probability:
        return maximum
    return policy_probability / behavior_probability


@dataclass(frozen=True, slots=True)
class OnePlyTarget:
    schema: str
    pre: ChessState
    action: ChessMove
    post: ChessState
    claim_draw: bool
    rollout_cutoff: bool
    target_source: str
    terminal_result: str | None
    terminal_termination: str | None
    mover_wdl: WDL
    advantage: float
    importance: float
    policy_probability: float
    behavior_probability: float
    importance_maximum: float


def build_one_ply_target(
    rules: PythonChessRules,
    pre: ChessState,
    action: ChessMove,
    *,
    online_pre_wdl: WDL,
    ema_post_wdl: WDL | None,
    policy_probability: float,
    behavior_probability: float,
    importance_maximum: float,
    claim_draw: bool,
    rollout_cutoff: bool,
) -> OnePlyTarget:
    """Validate the legal transition and freeze a mover-perspective TD(0) target.

    ``ema_post_wdl`` describes the player to move AFTER the action, not the
    mover. It is required only for a nonterminal child. A cutoff still requires
    bootstrap; a terminal child takes precedence over a simultaneous cutoff.
    The advantage uses expected score in [0,1], equal to half a Q=W-L change.
    These soft bootstrap labels are not observed game outcomes or replay v1.
    """
    if type(claim_draw) is not bool or type(rollout_cutoff) is not bool:
        raise ValueError("claim_draw and rollout_cutoff must be explicit booleans")
    current = validate_wdl(online_pre_wdl)
    importance = clipped_importance(
        policy_probability, behavior_probability, maximum=importance_maximum
    )
    if rules.outcome(pre, claim_draw=claim_draw) is not None:
        raise ValueError("cannot learn an action after a terminal position")
    mover = rules.view(pre).side_to_move
    post = rules.apply(pre, action)
    outcome = rules.outcome(post, claim_draw=claim_draw)
    if outcome is None:
        if ema_post_wdl is None:
            raise ValueError("nonterminal child requires explicit EMA bootstrap WDL")
        target = flip_wdl(ema_post_wdl)
        source = "ema-bootstrap"
    else:
        if ema_post_wdl is not None:
            raise ValueError("terminal child must use its observed result, not an EMA label")
        value = outcome.value_for(mover)
        target = (float(value == 1), float(value == 0), float(value == -1))
        source = "observed-terminal"
    return OnePlyTarget(
        schema=ONLINE_TARGET_SCHEMA,
        pre=pre,
        action=action,
        post=post,
        claim_draw=claim_draw,
        rollout_cutoff=rollout_cutoff,
        target_source=source,
        terminal_result=outcome.result.value if outcome else None,
        terminal_termination=outcome.termination if outcome else None,
        mover_wdl=target,
        advantage=expected_score(target) - expected_score(current),
        importance=importance,
        policy_probability=float(policy_probability),
        behavior_probability=float(behavior_probability),
        importance_maximum=float(importance_maximum),
    )
