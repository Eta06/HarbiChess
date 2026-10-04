"""Whole-opening colour-pair contrasts; capped games retain explicit uncertainty."""

import math
import random

import numpy as np


def _pairs(games):
    if not games or len(games) % 2:
        raise ValueError("complete nonempty colour pairs required")
    pairs, bounds, roots = [], [], set()
    for family in range(len(games) // 2):
        first, second = games[2 * family : 2 * family + 2]
        opening = tuple(first["opening"])
        if (
            first["opening_pair"] != family
            or second["opening_pair"] != family
            or first["candidate_color"] != "white"
            or second["candidate_color"] != "black"
            or tuple(second["opening"]) != opening
            or opening in roots
        ):
            raise ValueError("unique ordered source roots with white/black pairing required")
        roots.add(opening)
        values, limits = [], []
        for game in (first, second):
            score = game["score"]
            capped = game["termination"] == "max_plies"
            if score not in (0, 0.5, 1) or (capped and score != 0.5):
                raise ValueError("invalid score or unknown cap mislabelled")
            values.append(score)
            limits.append((0, 1) if capped else (score, score))
        pairs.append(sum(values) / 2)
        bounds.append(tuple(sum(limit[k] for limit in limits) / 2 for k in (0, 1)))
    return pairs, bounds


def compare(games, reference=None, *, seed=20261027, replicates=50000, alpha=0.05 / 3):
    """Direct score or matched family/colour score difference, never ply bootstrap.

    Default adjusted intervals implement three Bonferroni primary comparisons.
    Ordinary95% is descriptive. Neither interval resolves unknown cap outcomes.
    Linear empirical quantiles and Python Random whole-family sampling are fixed.
    """
    if type(replicates) is not int or replicates < 2 or not 0 < alpha < 1:
        raise ValueError("positive replication count and alpha in (0,1) required")
    values, bounds = _pairs(games)
    difference = reference is not None
    if difference:
        other, other_bounds = _pairs(reference)
        if len(games) != len(reference) or any(
            (game["opening_pair"], game["candidate_color"], game["opening"])
            != (ref["opening_pair"], ref["candidate_color"], ref["opening"])
            for game, ref in zip(games, reference, strict=True)
        ):
            raise ValueError("contrasts require identical source roots, order and colours")
        values = [value - ref for value, ref in zip(values, other, strict=True)]
        bounds = [
            (limit[0] - ref[1], limit[1] - ref[0])
            for limit, ref in zip(bounds, other_bounds, strict=True)
        ]
    count = len(values)
    rng = random.Random(seed)
    draws = [sum(rng.choices(values, k=count)) / count for _ in range(replicates)]
    mean = sum(values) / count
    radius = (2 if difference else 1) * math.sqrt(math.log(40) / (2 * count))
    lower = -1 if difference else 0
    return {
        "contrast": "matched score difference" if difference else "direct score",
        "families": count,
        "games_per_arm": len(games),
        "mean": mean,
        "pair_values": values,
        "bootstrap_pair_95": np.quantile(draws, [0.025, 0.975], method="linear").tolist(),
        "bootstrap_pair_adjusted": np.quantile(
            draws, [alpha / 2, 1 - alpha / 2], method="linear"
        ).tolist(),
        "adjusted_alpha": alpha,
        "hoeffding_pair_95": [max(lower, mean - radius), min(1, mean + radius)],
        "score_bounds_unknown_caps": [sum(b[k] for b in bounds) / count for k in (0, 1)],
        "replicates": replicates,
        "seed": seed,
        "quantiles": "linear empirical",
        "uncertainty_scope": (
            "Whole source-game colour pairs assumed independent, conditional frozen suite; "
            "caps use diagnostic0.5 in intervals, bounds resolve neither outcomes nor general Elo."
        ),
    }
