# MIHVER nonlinear invariant WDL preregistration

## Hypothesis

The decoupled auxiliary material head proves that current-board invariant features are present and
learnable (`MAE 0.00857`, `Pearson 0.99854`). The linear production WDL projection cannot represent
nonlinear phase/material interactions: mixed sampling improves natural micro CE but trades one
decisive class against the draw class, while macro CE continues improving at the final checkpoint.

Add one 64-unit nonlinear MLP over the same 20 current-board invariant features. Its final projection
is zero initialized, so release policy and WDL logits remain exact before training. The material head
stays separate and frozen during WDL transfer. No release trunk or policy parameter may train.

## Frozen experiment

- Exact release baseline and corrected trajectory-disjoint 148/48-game split.
- Stage-A deterministic auxiliary material gate unchanged.
- Mixed fixed-size WDL sampling unchanged: 32 outcome-balanced plus 32 natural game-balanced rows.
- Two arms: nonlinear global WDL; nonlinear global plus spatial value tower.
- Adam `5e-4`, 400 WDL steps, batch 64, validation every 20; unchanged seeds.
- Unchanged WDL gates: micro and macro CE improve by at least 0.10, Brier by at least 0.03,
  Pearson at least 0.20, adjacent outcome margins at least 0.03, ECE-10 at most 0.12.
- Auxiliary material predictions and release policy/trunk/value parameters must remain exact.

No threshold, exposure, run length, or checkpoint rule may change after results. Failure keeps
continuation ranking, Full Gumbel tactical qualification, continuous learning, and generation blocked.
