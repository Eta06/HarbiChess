# PUSULA powered retention preregistration

## Prior result and decision

`pusula-continuous-pilot-20260831-09` remains failed. It is not reclassified by
this protocol and none of its final qualification games enters the next gate.
The run accepted all three local updates and improved fresh CE, macro CE, Brier,
Pearson, policy imitation, tactical retention, and the 64-game search arena. It
nevertheless failed all five final historical non-inferiority intervals, fresh
ECE non-inferiority, and the continuation Spearman interval. Production
continuous generation and promotion remain disabled.

## Mechanism audit

The local historical tuning partition contains 29 games. Its 2,000-sample
paired bootstrap rejected only statistically established harm. Consequently an
inconclusive interval could select a checkpoint whose point deterioration was
already far outside the fixed practical margins. Update 1 selected residual
alpha `1.0`; its historical tuning CE moved from `0.87801` to `0.89735`, even
though the allowed cumulative deterioration is `0.003`. Later updates compounded
that error.

A diagnostic-only scaling sweep of the failed final residual did not alter the
verdict. Alpha `1.0` increased blind historical CE by `0.02427`; alpha `0.05`
instead changed it by `-0.00052`, retained fresh CE improvement `0.00673`, and
kept fresh ECE deterioration at `0.01154`. This supports a checkpoint-selection
bug rather than insufficient plastic representation.

The previous 32-position continuation sample had paired delta standard
deviation `0.28369`. A one-sided normal power calculation with alpha `0.05`,
power `0.80`, non-inferiority boundary `-0.020`, assumed effect `0`, 15% inflation,
and rounding to 32 requires 1,440 positions. The prior 32-position confidence
interval was therefore not adequately powered.

## Frozen replacement protocol

The new run is `pusula-continuous-pilot-20260831-10`, seed `2026090901`. It keeps
the PUSULA-09 architecture, policy targets, Full Gumbel budgets, replay sizes,
three updates, 40 steps per update, batch composition, learning rate, arena,
2,688 final phase-balanced attempts, minimum 744 known terminal games, and every
final cumulative margin unchanged.

Before any fresh result is visible, the following checkpoint-selection rules are
fixed:

- Every value checkpoint must satisfy both the point historical margins and the
  game-paired historical bootstrap safety test. An inconclusive interval cannot
  override a point estimate already beyond a practical margin.
- The historical margins remain CE `+0.003`, macro CE `+0.005`, Brier `+0.003`,
  Pearson `-0.010`, ECE `+0.010`, and absolute ECE `0.120` versus update-0 MIHVER.
- On the rolling game-disjoint fresh tuning partition, CE must improve while
  macro CE, Brier, and Pearson cannot regress versus update-0 MIHVER. Fresh ECE
  deterioration cannot exceed the existing `+0.020` final margin.
- Residual trust-region candidates remain deterministic and ordered. The prior
  grid is extended below `0.03125` with `0.015625` and `0.0078125`; no margin is
  changed. Among eligible candidates, minimum fresh CE, then macro CE, then the
  earliest step remains the selection rule.

Final continuation qualification uses a fresh, held-out, stratified set of
exactly 1,440 positions from the untouched historical validation trajectories.
The same release verifier, depth-4 deterministic oracle, and paired 20,000-sample
bootstrap are used for update-0 MIHVER and the final candidate. Its lower bound
must remain at least `-0.020`; verified-top agreement may lose at most one of the
1,440 positions. The 32-position set remains only a local diagnostic.

PUSULA-10 passes only if all local updates, exact checkpoint resume, policy,
material, tactical, arena, 1,440-position continuation, and the unchanged final
historical/fresh cumulative gates pass. A failure remains a failure. A pass may
authorize production continuous-loop integration, but never automatic release
promotion.
