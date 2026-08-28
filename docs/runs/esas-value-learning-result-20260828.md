# ESAS value-learning diagnostics result (2026-08-28)

## Decision

Both the short-horizon auxiliary experiment and the follow-up frozen-head causal
diagnostic failed. No trained model is accepted; learner/latest, generation,
arena, and promotion remain blocked.

The evidence now localizes the circular dependency:

- 256-simulation search is strong enough to beat raw policy;
- the current replay is too small and game-distribution-specific for value
  generalization;
- without a useful value head, 64/128 search cannot create a qualified teacher;
- without a qualified teacher, new replay remains blocked.

The next intervention is therefore search allocation, using literature-backed
FPU reduction to make flat-value low-budget search spend fewer simulations on
nearly every legal child. It changes no training target.

## Matched short-horizon experiment

Corrected artifact:
`artifacts/runs/esas-short-horizon-value-20260828-02/result.json`

The first artifact reported the pre-clip gradient norm under a gate worded for
the clipped norm. The metric was corrected and the exact frozen experiment was
rerun. This removed the gradient failure only; every substantive result was
unchanged.

| Metric | Baseline | Matched control | Auxiliary |
|---|---:|---:|---:|
| Validation WDL CE | 1.0989 | 1.9210 | 1.8980 |
| Outcome Pearson | +0.0851 | -0.0666 | -0.0730 |
| Value stddev | 0.0014 | 0.4210 | 0.3964 |
| Policy CE | 2.7716 | 2.6260 | 2.6263 |
| Tactical 128/256 | 7/7 | 5/6 | 5/6 |

The auxiliary head fit its own validation target better than the zero-weight
control, but neither arm learned a general WDL value. Both produced extreme,
anti-correlated value predictions and lost tactics. The shared trunk can optimize
policy on this replay while damaging value generalization.

## Frozen value-head-only causal diagnostic

Artifact:
`artifacts/diagnostics/esas-value-interference-20260828-01/bootstrap.json`

Freezing the policy and trunk did not solve the problem. The baseline step 0
remained the best checkpoint at WDL CE 1.0989; validation CE was already 1.1243
at step 79 and ended at 1.1494. This rejects the preregistered “joint gradient
interference is primary” branch.

KOPRU contains only 72 train games and 24 validation games. After max-ply masking,
the known-outcome game composition differs materially: train has 25 decisive and
18 draw games, while validation has 13 decisive and 4 draw games. Thousands of
rows do not create thousands of independent value labels because every position
in a game shares its eventual result. The effective value sample size is games,
not positions.

## Next causal branch

Do not retune auxiliary lambda, loss weight, duration, or learning rate on this
replay. Add opt-in parent-relative FPU reduction with separate root/interior
constants fixed from published engine practice, then rerun the unchanged system
teacher gate. If it qualifies 128/256 search, it unlocks genuinely larger fresh
replay and resolves the data side of the loop. If it fails, move to actual Full
Gumbel root/interior allocation rather than another learner experiment.
