"""Candidate-versus-champion arena quality estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArenaQuality:
    games: int
    wins: int
    draws: int
    losses: int
    score_rate: float
    elo_delta: float
    elo_low: float | None
    elo_high: float | None
    promotion_ready: bool


def _score_to_elo(score: float) -> float:
    bounded = min(1.0 - 1e-6, max(1e-6, score))
    return 400.0 * math.log10(bounded / (1.0 - bounded))


def estimate_arena_quality(
    wins: int,
    draws: int,
    losses: int,
    *,
    minimum_games: int = 200,
    promotion_elo: float = 0.0,
    z_score: float = 1.96,
) -> ArenaQuality:
    """Estimate Elo and a normal 95% interval from W/D/L arena outcomes.

    Each game contributes 1, 0.5, or 0 points. Promotion is conservative: the
    arena must meet its game budget and the interval's lower Elo bound must
    clear the configured threshold.
    """
    if min(wins, draws, losses, minimum_games) < 0:
        raise ValueError("arena counts and minimum_games must be non-negative")
    games = wins + draws + losses
    if games == 0:
        return ArenaQuality(0, wins, draws, losses, 0.5, 0.0, None, None, False)

    score_rate = (wins + 0.5 * draws) / games
    elo_delta = _score_to_elo(score_rate)
    if games == 1:
        elo_low = None
        elo_high = None
    else:
        squared_error = (
            wins * (1.0 - score_rate) ** 2
            + draws * (0.5 - score_rate) ** 2
            + losses * score_rate**2
        )
        standard_error = math.sqrt(squared_error / (games - 1) / games)
        elo_low = _score_to_elo(score_rate - z_score * standard_error)
        elo_high = _score_to_elo(score_rate + z_score * standard_error)

    promotion_ready = (
        games >= minimum_games
        and elo_low is not None
        and elo_low > promotion_elo
    )
    return ArenaQuality(
        games,
        wins,
        draws,
        losses,
        score_rate,
        elo_delta,
        elo_low,
        elo_high,
        promotion_ready,
    )
