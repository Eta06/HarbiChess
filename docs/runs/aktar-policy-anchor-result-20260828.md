# AKTAR Policy-Anchor Ablation Result

Date: 2026-08-28

## Decision

The frozen baseline-policy anchor matrix failed. No arm qualified for arena, and continuous
learning remains blocked. The rejected unanchored candidate was not reused.

## Frozen arms

| Anchor | Validation teacher CE | Top agreement | Baseline-policy KL | Raw tactics | Full Gumbel tactics | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 2.87061 | 8.85% | 0 | 1/8 | 4/8 | reference |
| 0.5 | 2.76604 | 16.15% | 0.08128 | 2/8 | 2/8 | failed |
| 1.0 | 2.76710 | 15.63% | 0.06833 | 2/8 | 2/8 | failed |
| 2.0 | 2.77556 | 14.58% | 0.04570 | 2/8 | 2/8 | failed |
| 4.0 | 2.79999 | 14.06% | 0.02110 | 2/8 | 2/8 | failed |

All four arms stopped at step 120 through the preregistered validation early-stopping rule.
Their non-policy parameter hashes, validation WDL logits, WDL loss, Brier score, Pearson
correlation, and ECE were exactly unchanged. Gradient norms were finite and below 1.0.

## Failure anatomy

Every arm retained both mate-in-one cases but lost both baseline-solved forced-defense cases.
At each failed root, Full Gumbel visited both legal root children 128 times. The candidate raw
policy selected the correct defense, yet search selected the losing alternative. Increasing the
anchor monotonically reduced average validation policy KL but did not change this outcome.

This rejects the hypothesis that a global mean policy anchor alone prevents the learner-transfer
regression. The next diagnostic must measure per-position and continuation-path policy drift and
determine whether sparse local drift or search allocation/value sensitivity causes the reversal.
The tactical or transfer thresholds are not relaxed after observing this result.

## Budget and continuation audit

A frozen follow-up sweep at 64, 128, 256, 512, and 800 simulations ruled out insufficient test
budget. The baseline retained both forced defenses at every budget, while all four anchor arms
missed both at every budget. The candidate raw policies correctly preferred both defenses.

After the losing `Ka1` continuation, the unchanged value head remained near zero. A small policy
shift made a stalemating king move the first interior expansion instead of a mate-in-one move.
Completed-Q then treated that observed draw as better than the near-zero value completion for
unvisited moves, and its visit-dependent min-max scaling concentrated later interior visits on the
draw line. More simulations reinforced rather than repaired the early allocation.

The implementation matches the reference Mctx Full Gumbel mechanism: deterministic interior
selection approximates `softmax(prior_logits + completed_q)`, and the default completed-Q transform
uses mixed-value completion plus visit-dependent scaling. The theoretical improvement guarantee is
conditional on correctly evaluated action values. HarbiChess's validation WDL is still close to
uninformative (`CE 1.09882`, versus `ln(3) = 1.09861`; Pearson `0.14224`), so that condition is not
currently reliable on these continuations.

This changes the diagnosis. The next decision is not another anchor-weight sweep. Either the
learner must improve value representation jointly while preserving qualified policy behavior, or
the exact-state Full Gumbel teacher must use an interior allocator whose exploration remains robust
under the currently weak value head. Both are foundational changes and require a new frozen
qualification rather than relaxing the failed transfer gate.

## Artifact

`artifacts/runs/aktar-policy-anchor-ablation-20260828-01/result.json`
