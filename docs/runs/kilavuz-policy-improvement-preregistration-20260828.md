# KILAVUZ Q-guided policy-improvement preregistration (2026-08-28)

## Hypothesis

Search-Q confidence and desired policy probability must be separate. A
KL-constrained mirror-descent update can turn stable 512/800 Q information into
a soft policy target without collapsing close-valued moves to one label or
moving uncertain branches arbitrarily.

## Frozen target rule

For every VERI row, start from the raw network legal policy. An action is
Q-qualified only when the existing uncertainty label gives it nonzero
confidence. Qualified actions receive multiplier `exp((Q - anchor) / tau)`;
unqualified actions retain multiplier 1. `anchor` is the raw-policy-weighted
mean qualified Q. `tau` is found deterministically so target-to-raw KL is at
most `0.10`; no target may exceed that trust region. The result is normalized
over all legal actions. Q values that are close therefore remain close in
probability, while uncertain actions preserve their raw prior rather than being
forced to zero.

No learner runs before this target passes on the frozen VERI split:

1. validation verified expected-value gain over raw has a positive 95% paired
   bootstrap lower bound;
2. mean target-to-raw KL is at most 0.10;
3. rows with verified expected-value loss of at least 0.025 are at most 10%;
4. target-top-action harmful ratio is at most 10% and mean verified regret is
   at most 0.10;
5. target effective-action count is at least 50% of raw on average;
6. at least 95% of source rows remain labelable.

Dataset and thresholds are frozen before target results. Passing authorizes an
unchanged 480-step, batch-16 rank-8 policy-transfer ablation with a fresh seed;
it does not authorize arena, generation, or promotion.
