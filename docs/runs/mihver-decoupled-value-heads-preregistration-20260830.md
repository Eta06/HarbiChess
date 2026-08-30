# MIHVER decoupled auxiliary and WDL heads preregistration

## Architectural correction

The retained WDL arm passed micro CE, Brier, expected-score correlation, ordered margins, ECE, and
material retention, but failed macro CE. The previous design forces one WDL distribution to serve
two different semantics:

- deterministic material is an auxiliary representation test;
- terminal WDL is the production game-outcome prediction.

Requiring production expected score to remain numerically equal to material prevents confident
decisive WDL calibration. The correct architecture keeps the material capability in a separate
auxiliary scalar head and gives production WDL independent zero-initialized global and spatial
residuals.

## Function-preserving network

- Preserve the entire release policy/trunk/legacy-WDL path exactly.
- Add a count-scaled 20-feature auxiliary material scalar head.
- Keep the zero-initialized 20-feature production-WDL residual.
- Keep the zero-initialized independent spatial production-WDL tower.
- Initial policy and production WDL logits must be bitwise equal to release.

## Stage A: auxiliary material gate

- Train only the auxiliary scalar head with deterministic material MSE.
- Same trajectory-disjoint 8,192/4,096 positions, Adam `2e-3`, batch 64, 200 steps, seed
  `2026083061`.
- Same hard gates: 50% MSE reduction, Pearson at least 0.80, MAE at most 0.05.
- All release and production-WDL parameters remain frozen and hash-exact.

## Stage B: production WDL gate

Runs only if Stage A passes. Start from the qualified auxiliary model and compare:

1. `global-wdl`: train only the 20-feature production-WDL residual.
2. `global-tower-wdl`: train the production-WDL residual and independent spatial tower.

- Same deduplicated corrected 148/48 terminal-game split; unknown/max-ply excluded.
- Outcome/game-balanced batches of 64, Adam `5e-4`, 400 steps, validation every 20, seed
  `2026083073`.
- Auxiliary material output, release parameters, and policy logits remain exact.
- Existing WDL hard gates remain unchanged: micro/macro CE improvement at least 0.10, Brier
  improvement at least 0.03, Pearson at least 0.20, adjacent outcome margins at least 0.03,
  ECE-10 at most 0.12.

Only Stage B passing authorizes continuation action-value ranking and then Full Gumbel tactical
retention. Continuous learning, generation, arena, and promotion remain blocked.

