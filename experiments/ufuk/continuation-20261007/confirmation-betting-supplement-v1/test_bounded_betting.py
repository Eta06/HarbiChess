"""Synthetic arrays/exact null enumeration only; no actual chess/model data."""

import itertools
import math
import random

import pytest
from bounded_betting import (
    ALPHA_EACH,
    FRACTIONS,
    METRICS,
    eight_primary_bounds,
    exact_rejected,
    log_capital,
    lower_confidence_bound,
)


def test_boundaries_zero_capital_and_allwins_analytic_inversion():
    assert lower_confidence_bound([0.] * 48, 0, 1) == 0
    assert lower_confidence_bound([-1.] * 48, -1, 1) == -1
    assert math.isfinite(log_capital([0.] * 48, 0.5, 0, 1))  # f=1 term is zero.
    b = lower_confidence_bound([1.] * 48, 0, 1)
    assert 0 < b < 1
    assert exact_rejected([1.] * 48, b, 0)
    expected = sum(((1 - f) + f / b) ** 48 for f in FRACTIONS) / 10
    assert expected == pytest.approx(1 / ALPHA_EACH, rel=1e-10)
    with pytest.raises(ValueError):
        log_capital([0.] * 48, 0., 0, 1)


def test_monotone_capital_affine_support_and_empirical_mean_ceiling():
    xs = [0., 0.5, 1.] * 16
    logs = [log_capital(xs, mu, 0, 1) for mu in (0.1, 0.2, 0.4, 0.5, 0.9)]
    assert all(a >= b for a, b in itertools.pairwise(logs))
    assert logs[3] <= 1e-12
    b = lower_confidence_bound(xs, 0, 1)
    assert 0 <= b <= 0.5
    assert lower_confidence_bound([2 * x - 1 for x in xs], -1, 1) == pytest.approx(2 * b - 1)


def test_input_validation_and_exact_eight_gate_tightening():
    for xs in ([1.] * 47, [math.nan] * 48, [1.1] * 48, [math.inf] * 48):
        with pytest.raises(ValueError):
            lower_confidence_bound(xs, 0, 1)
    metrics = {k: [1.] * 48 for k in METRICS}
    result = eight_primary_bounds({20262905: metrics, 20262906: metrics})
    assert len(result["records"]) == 8 and result["passed"]
    assert result["union_bound"] == 0.00625
    assert result["original_bootstrap_still_required"]
    with pytest.raises(ValueError):
        eight_primary_bounds({20262905: metrics})


def test_heterogeneous_independent_mean_null_amgm_exact_enumeration():
    # One E-factor expectation can exceed1, while the full-product average-null
    # expectation stays<=1. This is FIXED-n, not a martingale for optional stopping.
    probs, mu = (0.1, 0.8, 0.3), 0.4
    for f in FRACTIONS:
        expectation = 0.
        for xs in itertools.product((0., 1.), repeat=3):
            probability = math.prod(p if x else 1 - p for x, p in zip(xs, probs, strict=False))
            capital = math.prod(1 + f * (x - mu) / mu for x in xs)
            expectation += probability * capital
        assert expectation <= 1 + 1e-12


def test_exact_binomial_null_and_fixed_simulation_not_coverage_claim():
    threshold = -math.log(ALPHA_EACH) + 1e-12
    rejected = {}
    for mu in (0.1, 0.5, 0.9):
        ks = [k for k in range(49)
              if log_capital([1.] * k + [0.] * (48 - k), mu, 0, 1) >= threshold]
        mass = math.fsum(math.comb(48, k) * mu**k * (1 - mu)**(48 - k) for k in ks)
        assert mass <= ALPHA_EACH + 1e-13
        rejected[mu] = set(ks)
    rng = random.Random(2026100701)
    successes = sum(rng.getrandbits(48).bit_count() in rejected[0.5] for _ in range(10000))
    assert successes / 10000 <= ALPHA_EACH + 0.003
