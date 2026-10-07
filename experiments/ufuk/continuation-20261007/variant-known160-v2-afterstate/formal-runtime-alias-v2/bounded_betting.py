"""Fixed-n independent-root mean LCBs; no adaptive bets or optional stopping."""

import math
from fractions import Fraction

ALPHA_EACH = 0.00078125
FRACTIONS = tuple(i / 10 for i in range(1, 11))
ROOTS = 48
# Fixed prospective upward rejection margin; protects near-threshold floating error.
LOG_REJECTION_MARGIN = 1e-12
BISECTION_STEPS = 100
METRICS = ("E8_direct", "SF_gain_E8", "parent_direct", "SF_gain_parent")


def checked(values, lower, upper):
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError("finite ordered bounds")
    xs = tuple(float(x) for x in values)
    if len(xs) != ROOTS or any(not math.isfinite(x) or not lower <= x <= upper for x in xs):
        raise ValueError("exact48 finite independent rootblock outcomes within bounds")
    return xs


def log_add(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(min(a, b) - m))


def log_capital(values, mu, lower, upper):
    """log mean_f product_i [(1-f)+f*(x_i-L)/(mu-L)], for mu>L."""
    xs = checked(values, lower, upper)
    if not math.isfinite(mu) or not lower < mu <= upper:
        raise ValueError("null mean must be strictly above lower support boundary")
    logden = math.log(mu - lower)
    logs = []
    for f in FRACTIONS:
        terms = []
        for x in xs:
            a = -math.inf if f == 1 else math.log1p(-f)
            b = -math.inf if x == lower else math.log(f) + math.log(x - lower) - logden
            terms.append(log_add(a, b))
        logs.append(-math.inf if -math.inf in terms else math.fsum(terms))
    m = max(logs)
    if m == -math.inf:
        return m
    return m + math.log(math.fsum(math.exp(x - m) for x in logs)) - math.log(len(logs))


def exact_rejected(values, mu, lower):
    """Exact rational final certification for represented scores/bound (no rounding claim)."""
    denominator = Fraction(mu) - Fraction(lower)
    if denominator <= 0:
        return False
    capital = Fraction(0)
    for i in range(1, 11):
        f = Fraction(i, 10)
        product = Fraction(1)
        for x in values:
            product *= 1 - f + f * (Fraction(x) - Fraction(lower)) / denominator
        capital += product / 10
    return capital >= 1280  # EXACT alpha=1/1280; decimal float is not exact.


def lower_confidence_bound(values, lower, upper):
    xs = checked(values, lower, upper)
    mean = math.fsum(xs) / ROOTS
    if mean == lower:
        return lower
    lo, hi = lower, mean
    threshold = -math.log(ALPHA_EACH) + LOG_REJECTION_MARGIN
    # At empirical mean AM-GM gives E<=1. Always return rejected-side bracket,
    # never the nonrejected upper bracket (which would inflate the numerical LCB).
    for _ in range(BISECTION_STEPS):
        mid = lo + (hi - lo) / 2
        if mid in (lo, hi):
            break
        if log_capital(xs, mid, lower, upper) >= threshold:
            lo = mid
        else:
            hi = mid
    # Logs locate the crossing; exact arithmetic certifies the reported endpoint.
    # If floating approximation overshoots, step downward; fail conservatively to L.
    for _ in range(64):
        if exact_rejected(xs, lo, lower):
            return lo
        lo = math.nextafter(lo, lower)
        if lo <= lower:
            return lower
    return lower


def eight_primary_bounds(seed_metrics):
    """Input fixed TWO seed keys, each exactly four adverse-cap48-root arrays."""
    if len(seed_metrics) != 2:
        raise ValueError("both fixed preregistered seed recipes required")
    records = []
    for seed, metrics in seed_metrics.items():
        if set(metrics) != set(METRICS):
            raise ValueError("exact four required inferential metrics perseed")
        for name in METRICS:
            lower, upper = (0., 1.) if name.endswith("direct") else (-1., 1.)
            xs = checked(metrics[name], lower, upper)
            threshold = 0.5 if name.endswith("direct") else 0.
            bound = lower_confidence_bound(xs, lower, upper)
            records.append(dict(seed=seed, metric=name, support=[lower, upper],
                                point=math.fsum(xs) / ROOTS, lower=bound,
                                required_lower_gt=threshold, passed=bound > threshold))
    return dict(schema="ONE-confirmation-fixed-root-betting-tightening-v1",
                records=records, passed=all(r["passed"] for r in records),
                alpha_each=ALPHA_EACH, union_bound=8 * ALPHA_EACH,
                fraction_grid=list(FRACTIONS), root_blocks=ROOTS,
                original_bootstrap_still_required=True,
                claim=("fixed-n conditional finite-sample bound "
                       "under independent bounded rootblocks"),
                numerical_certificate="exact-rational final E>=1280 for every LCB>L")
