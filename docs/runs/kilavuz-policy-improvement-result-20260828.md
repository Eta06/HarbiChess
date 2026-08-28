# KILAVUZ Q-guided policy-target result (2026-08-28)

## Decision

KILAVUZ failed one preregistered guardrail. No learner, arena, generation, or
promotion is authorized from this artifact.

## Evidence

The KL-constrained target preserved a broad distribution: validation mean
effective-action ratio was 90.50% of raw and target-to-raw KL was 0.10. Its
verified expected-value gain was positive with a 95% interval of +0.00887 to
+0.01657, and no validation row had expected-value loss of at least 0.025.

The target-top harmful-action ratio was 11/95 = 11.58%, exceeding the frozen
10% limit. The threshold is unchanged and the experiment is rejected. The
average cross-budget Q used for improvement can still make a riskier action the
target leader even when the full soft distribution has positive expected
value. A separate preregistered experiment may replace only that statistic with
the conservative `min(Q512, Q800)` estimate; it must pass the same gates.

## Frozen artifact

- `artifacts/diagnostics/kilavuz-policy-target-20260828-01/targets.json`
