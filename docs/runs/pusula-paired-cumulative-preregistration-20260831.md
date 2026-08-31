# PUSULA paired cumulative preregistration

## Prior result and hypothesis

`pusula-continuous-pilot-20260831-14` remains failed and is not reclassified.
Updates 1 and 2 passed, but update 3 rolled back because every positive residual
interpolation crossed the legacy initial-MIHVER point-estimate CE limit. The
smallest alpha missed that limit by about `0.000002`, while its 29-game paired
interval remained inconclusive rather than demonstrating harm. The failure
exposed a protocol mismatch: a hard point estimate still overruled the
preregistered game-paired non-inferiority design.

PUSULA-15 tests the correction on entirely fresh replay and seeds. It does not
change the model, training duration, exposure, search allocator, policy target,
or practical deterioration/improvement margins.

## Frozen statistical decision

- Complete games are the independent unit. Positions from one game stay in the
  same game-cluster bootstrap sample.
- Local historical tuning uses one-sided 95% paired bootstrap intervals. Its 29
  games are a safety screen only: reject a checkpoint only when the lower bound
  proves deterioration beyond the frozen margin. An inconclusive interval may
  continue to the powered sealed qualification but can never authorize
  production.
- The old 48-game historical holdout is retained as a diagnostic only. It has
  no hard point-estimate veto and cannot authorize production.
- The final sealed old-distribution qualification is the only old-capability
  production gate: 1,536 phase-balanced attempts, at least 384 known terminal
  games, paired initial-MIHVER versus final-candidate predictions, and 20,000
  bootstrap samples.
- The existing ECE power calculation is the binding old-capability plan: with
  SD `0.070`, one-sided alpha `0.05`, power `0.80`, margin `0.010`, zero assumed
  deterioration, and 15% inflation, about 350 known games are required. The
  fixed floor of 384 exceeds it. Other old endpoints require fewer games.
- Final fresh qualification remains 2,688 attempts with at least 744 known
  games and the same 20,000-sample game-paired bootstrap.

All co-primary margins remain frozen:

| Endpoint | Final pass condition |
|---|---|
| old WDL CE | candidate-minus-MIHVER upper bound `<= +0.003` |
| old macro WDL CE | upper bound `<= +0.005` |
| old Brier | upper bound `<= +0.003` |
| old expected-score Pearson | lower bound `>= -0.010` |
| old ECE-10 | upper bound `<= +0.010`; candidate absolute `<= 0.120` |
| fresh WDL CE | MIHVER-minus-candidate lower bound `>= +0.002` |
| fresh macro WDL CE | lower bound `>= 0.000` |
| fresh Brier | lower bound `>= 0.000` |
| fresh expected-score Pearson | lower bound `>= 0.000` |
| fresh ECE-10 | lower bound `>= -0.020`; candidate absolute `<= 0.150` |

## Frozen run

- Run: `pusula-continuous-pilot-20260831-15`
- Seed: `2026091401`
- Initial model, MIHVER stable base, plastic residual, historical weight `2.0`,
  3 updates, 40 steps/update, batch 64, learning rate `1e-4`, residual alpha
  grid, 768 self-play attempts/update, minimum 192 known games/update, rolling
  two-generation replay, Full Gumbel targets/search, policy/tactical gates,
  1,440-position continuation gate, 64-game final arena, integrity checks, and
  exact-resume checks are unchanged from PUSULA-14.
- PUSULA-14 replay, checkpoints, tuning outcomes, and arena results are excluded
  from PUSULA-15 training, selection, and qualification.
- Passing may authorize production continuous-loop integration. It never
  promotes a release checkpoint automatically.
