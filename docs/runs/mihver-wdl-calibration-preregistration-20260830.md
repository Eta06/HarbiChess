# MIHVER WDL calibration preregistration

## Qualified starting point

`mihver-count-scaled-material-20260830-01` passed the deterministic gate with the selected
`global-linear` arm:

- initialization policy/WDL delta `0.0` and exact release parameter hash;
- held-out material MSE `0.00001774`, MAE `0.002992`, Pearson `0.999702`.

The selected model is not a candidate. It only authorizes this WDL experiment.

## Measured loss interaction

On the first deterministic outcome/game-balanced batch, the WDL gradient norm over the new value
branches is `2.40609`, material-retention norm is `0.006784`, ratio `354.65`, and cosine `-0.145`.
An equal-weight auxiliary loss would be ineffective. No post-result weight sweep is allowed.

## Frozen arms

Both arms start from the selected material-qualified model. The release policy/trunk/legacy-value
parameters and the qualified global-linear material anchor remain frozen; only the independent
value tower is trainable.

1. `tower-wdl`: terminal WDL cross-entropy only.
2. `tower-wdl-retained`: terminal WDL cross-entropy plus `350 * material MSE`, fixed from the
   measured gradient ratio.

## Frozen data and schedule

- Same deduplicated schema 10-12 corrected replay pool from the exact release baseline.
- Unknown/max-ply games excluded from WDL.
- Same trajectory-disjoint 148 train / 48 validation games, zero fingerprint overlap.
- Outcome- and game-balanced training batches of 64.
- Adam `5e-4`, 400 steps, validation every 20, seed `2026083073`.
- No search, policy target, arena, generation, or policy parameter update.

## Hard gates

At least one checkpoint must satisfy all conditions simultaneously:

- material probe remains qualified: MSE at least 50% below release, Pearson at least 0.80,
  MAE at most 0.05;
- validation micro and macro WDL CE improve by at least 0.10 over release;
- validation WDL Brier improves by at least 0.03;
- validation expected-score Pearson at least 0.20;
- ordered loss/draw/win means with both adjacent margins at least 0.03;
- validation ECE-10 at most 0.12;
- release and global-anchor parameter hashes remain exact.

Select the passing checkpoint with lowest validation macro WDL CE. Prefer `tower-wdl` if both arms
pass within `0.01` macro CE; otherwise select the lower-CE arm.

Only a full pass authorizes continuation action-value ranking. Continuous learning, generation,
arena, and promotion remain blocked.

