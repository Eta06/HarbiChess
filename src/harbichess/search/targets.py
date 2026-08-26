"""Policy-target extraction that keeps exploration noise out of supervision."""

from __future__ import annotations

from collections.abc import Mapping

from harbichess.core.state import ChessMove
from harbichess.search.mcts import SearchResult


def visit_policy(search: SearchResult) -> tuple[tuple[ChessMove, float], ...]:
    """Return the normalized policy for visited root actions."""

    visited = tuple(item for item in search.moves if item.visits > 0)
    total = sum(item.visits for item in visited)
    if total <= 0:
        raise ValueError("policy targets require at least one root visit")
    return tuple((item.move, item.visits / total) for item in visited)


def prune_noise_attributable_visits(
    search: SearchResult,
    clean_priors: Mapping[ChessMove, float],
) -> tuple[tuple[ChessMove, float], ...]:
    """Remove visits attributable only to a positive root-noise prior boost.

    This is deliberately conservative: the visit leader is never pruned and every
    other visited action retains one visit. It is a qualification transform, not a
    default training target, until measured strength supports enabling it.
    """

    visited = tuple(item for item in search.moves if item.visits > 0)
    total = sum(item.visits for item in visited)
    if total <= 0:
        raise ValueError("policy target pruning requires root visits")
    if any(item.move not in clean_priors for item in visited):
        raise ValueError("clean priors must cover every visited action")

    leader = max(visited, key=lambda item: (item.visits, item.move.uci))
    retained: list[tuple[ChessMove, int]] = []
    for item in visited:
        removable = 0
        if item is not leader:
            positive_noise_boost = max(0.0, item.prior - clean_priors[item.move])
            removable = min(item.visits - 1, round(total * positive_noise_boost))
        retained.append((item.move, item.visits - removable))
    retained_total = sum(visits for _, visits in retained)
    return tuple((move, visits / retained_total) for move, visits in retained)
