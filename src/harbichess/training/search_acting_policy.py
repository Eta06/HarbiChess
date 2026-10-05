"""Explicit full-support greedy acting mixture; search supervision stays separate."""

from __future__ import annotations

import math
from dataclasses import dataclass

from harbichess.training.ownsearch_targets import OwnSearchConfig

LEGACY_LEDGER = "pre-action-masked-search-behavior-v4"
GREEDY_LEDGER = "pre-action-greedy-mixture-search-behavior-v5"
LEGACY_BEHAVIOR = "raw-T1-except-preselected-search-policy-with-visited-mate1-loss-shield-T1;no-PPO"
GREEDY_BEHAVIOR = (
    "raw-T1-except-preselected-fullsupport-selected-action-search-policy-mixture-T1;no-PPO"
)


@dataclass(frozen=True, slots=True)
class GreedyOwnSearchConfig(OwnSearchConfig):
    """Optional versioned acting semantics; existing six-field configs stay unchanged."""

    greedy_fraction: float

    def __post_init__(self):
        OwnSearchConfig.__post_init__(self)
        if (
            type(self.greedy_fraction) not in (float, int)
            or not math.isfinite(self.greedy_fraction)
            or not 0 <= self.greedy_fraction < 1
        ):
            raise ValueError("greedy acting fraction must be finite in [0,1)")


def search_config_from_dict(value):
    cls = GreedyOwnSearchConfig if "greedy_fraction" in value else OwnSearchConfig
    return cls(**value)


def ledger_semantics(config):
    if isinstance(config, GreedyOwnSearchConfig):
        return GREEDY_LEDGER, GREEDY_BEHAVIOR
    return LEGACY_LEDGER, LEGACY_BEHAVIOR


def mixture_policy(search_policy, selected_index, fraction):
    """Mix a legal selected point mass with shielded search pi, retaining its support."""
    if (
        type(selected_index) is not int
        or not 0 <= selected_index < len(search_policy)
        or type(fraction) not in (float, int)
        or not math.isfinite(fraction)
        or not 0 <= fraction < 1
        or not search_policy
        or any(not math.isfinite(p) or p < 0 for p in search_policy)
        or not math.isclose(math.fsum(search_policy), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("invalid selected action, normalized search policy or mixture")
    # Zero fraction is the exact legacy path, not a second normalization.
    if fraction == 0:
        return tuple(search_policy)
    row = tuple(
        (1 - fraction) * p + (fraction if i == selected_index else 0)
        for i, p in enumerate(search_policy)
    )
    if any(p > 0 and q == 0 for p, q in zip(search_policy, row, strict=True)):
        raise ValueError("greedy mixture underflow removed search-policy support")
    total = math.fsum(row)
    return tuple(p / total for p in row)
