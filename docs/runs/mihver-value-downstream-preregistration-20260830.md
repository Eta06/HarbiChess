# MIHVER value downstream qualification preregistration

## Frozen candidate

- Source result: `mihver-nonlinear-wdl-20260830-01`.
- Candidate: preregistered selected `global-wdl` checkpoint, SHA-256
  `6c535585e952b3d8ba5ff2331fe4dc992d94c586fdfa6a7c3c0cbc9e2d229dbb`.
- Baseline: exact KOPRU release checkpoint.
- Training is closed. This stage only evaluates the frozen candidate.

## Convention correction

Continuation ranking must consume the network's public WDL forward result. The legacy diagnostic
called the release-only private `_value_logits(_trunk(...))` path, which bypasses all new residual
value heads. Correcting that evaluator is required before measurement and changes no search or
training behavior.

## Continuation action-value gate

- Select 32 records by deterministic stratification from the existing trajectory-disjoint validation
  games, seed `2026083091`.
- Score every legal child with full baseline/candidate WDL forward passes and a deterministic depth-4
  tactical oracle.
- Candidate mean Spearman must improve by at least 0.05 and remain positive.
- Candidate verified-top agreement must not regress.

## Full Gumbel tactical retention gate

- Existing deterministic tactical suite, Full Gumbel, 256 simulations, 8 workers, zero Gumbel noise,
  seed `2026082883`.
- Candidate must solve at least 4/8 cases and may not lose any baseline-solved case.
- Raw policy solve count and exact policy logits must not regress.

Both gates must pass. Thresholds, positions, seeds, and budgets stay frozen after results. Continuous
learning and generation remain blocked even on a pass until the complete value qualification result
is reviewed.
