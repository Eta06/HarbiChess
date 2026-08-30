# MIHVER count-scaled invariant value preregistration

## Finding from the first invariant probe

The first global invariant arm preserved release logits and parameters exactly, reached validation
Pearson `0.8756`, and reduced MSE from `0.02335` to `0.00927`, but failed the unchanged MAE gate at
`0.07293`. It is rejected. The learning curve was still improving at step 200, but this follow-up
does not extend exposure.

The 104-plane mean mixes current and seven historical piece snapshots and scales piece counts by
`1/64`. That makes the simple deterministic target unnecessarily ill-conditioned and exposes
irrelevant history. The follow-up changes representation, not thresholds or duration.

## Frozen representation change

The direct invariant residual receives exactly 20 features:

- sums of the current 12 canonical piece planes, producing actual own/opponent piece counts;
- means of the 8 metadata planes, whose values are already normalized.

The independent spatial value tower is unchanged. Both residual outputs remain zero-initialized;
release policy and WDL logits must again be bitwise exact before training, and all release
parameters remain frozen.

## Frozen experiment

- Same trajectory-disjoint 8,192 train / 4,096 validation material-probe positions.
- Same global-linear and invariant-tower arms.
- Same Adam `2e-3`, batch 64, 200 steps, validation every 20, seed `2026083061`.
- Same hard gates: 50% MSE reduction, Pearson at least 0.80, MAE at most 0.05, exact initialization
  and frozen release-parameter hash.
- Same selection rule: prefer the global-linear arm unless the passing tower has at least 20%
  lower MSE.

No WDL/RL stage is authorized unless this fresh run passes every frozen gate.

