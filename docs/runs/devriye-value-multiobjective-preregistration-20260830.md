# DEVRIYE value multi-objective loss preregistration

## Diagnosis

The pooled 88-game reservoir improved fresh held-out CE for every tested batch,
but no arm passed all exact old/fresh gates because expected-score Pearson moved
slightly backward and at least one old CE metric regressed. More replay or a
larger batch with the same pointwise CE objective is rejected.

## Frozen cached ablation

- Reuse the exact pooled three-generation fit/held-out split.
- Use one value-only Adam step, learning rate `1e-4`, total batch 1024 split
  equally between historical and pooled fresh replay.
- For each half, compute both natural micro WDL CE and equal-class macro WDL CE.
- Add a differentiable expected-score Pearson loss using
  `P(win) - P(loss)` against outcome values `-1, 0, 1` in each half.
- Test Pearson-loss weights `0`, `0.25`, `1`, and `4`. The CE terms and every
  other setting remain fixed. No post-result weight is allowed.
- Report gradient norm and old fixed-validation plus pooled fresh held-out
  micro/macro CE, Pearson, ECE, and outcome margins.

## Gate

The existing exact multi-domain gate remains unchanged: old micro CE, macro CE,
and Pearson cannot regress; fresh held-out micro CE must improve and fresh
Pearson cannot regress; margins and numerical safety must hold.

Choose the smallest passing Pearson weight. If none passes, reject this objective
and move to value representation/target changes. Cached evidence can only select
a mechanism for a later fresh qualification.
