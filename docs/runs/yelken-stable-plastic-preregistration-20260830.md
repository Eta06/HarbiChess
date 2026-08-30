# YELKEN stable-base plus plastic-residual preregistration

## Hypothesis

DEVRIYE's mutable MIHVER value heads pass each local update but accumulate small
WDL and continuation-ranking drift. A function-preserving, independently
trainable residual value pathway should absorb fresh replay while the qualified
MIHVER value representation remains a stable anchor.

## Frozen inputs and controls

- Start every arm from the qualified MIHVER `global-wdl` checkpoint with SHA-256
  `6c535585e952b3d8ba5ff2331fe4dc992d94c586fdfa6a7c3c0cbc9e2d229dbb`.
- Reuse the three completed DEVRIYE `-13` replay generations only for the cached
  ablation; keep their deterministic game split, row order, sampler seeds,
  optimizer, learning rate, batch composition, step count, and validation sets
  identical across arms.
- Do not alter Full Gumbel, target generation, policy loss, policy architecture,
  self-play, arena, or promotion rules in this experiment.
- The plastic branch must be zero-output initialized. Before training, policy and
  WDL logits must match MIHVER exactly within `1e-6` absolute error.

## Controlled arms

1. `frozen-base`: freeze every MIHVER value parameter and train only the new
   plastic residual at the normal value learning rate.
2. `low-lr-base`: train the plastic residual at the normal rate and MIHVER value
   parameters with gradients scaled to `0.1x`.
3. `mutable-base-control`: train the plastic residual and MIHVER value parameters
   at the same rate. This is the high-plasticity control, not the default.

The policy head follows the unchanged DEVRIYE policy regime but is composed from
the same policy checkpoint in every arm. Therefore value-arm selection cannot be
won by a policy change.

## Cached ablation gates

Select the earliest preregistered checkpoint that satisfies all of the following;
do not change thresholds or sample size after seeing results:

- old fixed-validation WDL micro CE and macro CE do not exceed the MIHVER start;
- old fixed-validation expected-score Pearson does not fall below the start;
- fresh held-out WDL micro CE and macro CE do not exceed its start, and Pearson
  does not fall below its start;
- both WDL outcome margins remain at least `0.03`, and ECE-10 remains at most
  `0.12`;
- continuation mean Spearman and verified-top agreement do not regress from the
  MIHVER start;
- Full Gumbel tactical solve count does not regress;
- policy logits and frozen material predictions remain unchanged by value-only
  selection.

If multiple arms pass, choose frozen-base first, then low-lr-base, then the
mutable control. This explicitly prefers the least mutable qualified base.

## Fresh continuous pilot

Only a cached-ablation winner may enter a fresh three-update rolling-replay pilot.
Use the existing DEVRIYE compute, search, data-balance, local safety gates, and
fresh seeds. In addition to every per-update gate, the final checkpoint must not
regress versus the original MIHVER start on WDL micro CE, macro CE, Pearson,
continuation Spearman, continuation verified-top agreement, or Full Gumbel
tactical solve count. The final paired search score must remain at least `0.50`.

Production continuous generation is authorized only if all three updates and all
cumulative gates pass. A failed arm or pilot remains diagnostic and cannot be
promoted.
