# KOK joint representation transfer result (2026-08-28)

## Decision

KOK failed its frozen internal holdout gate. It emitted no checkpoint and did
not authorize external validation, search qualification, arena, generation,
or promotion.

## Result

All arms preserved broad replay behavior well inside the frozen limits, but
none transferred SIPER to the 59-position unseen-game holdout.

| Policy anchor | Gap closure | Policy KL | WDL KL | Harmful | Verified 95% lower |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -10.03% | 0.00558 | 0.00000095 | 15.25% | -0.00959 |
| 4 | -6.02% | 0.00221 | 0.00000090 | 13.56% | -0.00961 |
| 16 | -3.78% | 0.00060 | 0.00000076 | 16.95% | -0.01906 |

Expected-score drift was below 0.00072 in every arm and gradients remained
well below norm 5.0. The failure is therefore not WDL collapse, numerical
instability, or unconstrained broad policy drift.

## Interpretation

Opening the release trunk did not repair the CIPA generalization failure. This
rejects frozen-adapter capacity as the primary explanation. The evidence now
points to high-budget label sparsity and game correlation: 320 fit positions
cannot identify a policy update that generalizes to new games. The next test
must increase labels from the already generated replay, not learner exposure
or self-play generation.

Artifact: `artifacts/diagnostics/kok-joint-policy-20260828-01/result.json`
