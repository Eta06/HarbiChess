# MIHVER distributional material preregistration

## Root-cause finding

The first material probe optimized only expected score `P(win) - P(loss)`. That objective does not
identify the draw probability: many incompatible WDL distributions have the same expected score.
The selected material model passed scalar metrics but began terminal WDL transfer with draw-class
CE `3.1025` and macro CE `1.4932`.

The subsequent tower learned useful held-out outcome ordering (Pearson `0.318`, margins
`0.0584/0.0564`, Brier improvement `0.0487`) but could not simultaneously undo this initial
miscalibration, preserve the material MAE gate, and clear the unchanged CE gates. That run is
rejected.

## Frozen correction

For deterministic material value `v` in `[-1, 1]`, use the fully identified soft target:

- win: `max(v, 0)`;
- draw: `1 - abs(v)`;
- loss: `max(-v, 0)`.

Train with soft-target WDL cross-entropy rather than expected-score MSE. Evaluation keeps the same
scalar expected-score metrics and hard gates, so this is a target-structure correction rather than
a relaxed success criterion.

## Frozen experiment

- Same count-scaled 20-feature invariant representation and zero-initialized residuals.
- Same bitwise initialization and frozen release-parameter requirements.
- Same global-linear and invariant-tower arms.
- Same trajectory-disjoint 8,192 train / 4,096 validation positions.
- Same Adam `2e-3`, batch 64, 200 steps, validation every 20, seed `2026083061`.
- Same hard gates: 50% MSE reduction, Pearson at least 0.80, MAE at most 0.05.
- Additionally record soft-target CE and draw-probability MAE; these are diagnostic and do not
  replace the frozen hard gates.
- Same selection rule preferring the simpler passing global arm unless the tower has at least 20%
  lower MSE.

Only a full pass may restart the already-preregistered WDL calibration arms from this newly
qualified model. No continuous learning, generation, arena, or promotion is authorized.

