# MIHVER mixed WDL sampling preregistration

## Measured failure mode

The decoupled global+tower WDL arm passed macro CE improvement (`1.09890 -> 0.99577`), outcome
ordering, ECE, and auxiliary isolation. It missed micro CE improvement by `0.02013` and Brier by
`0.00148`.

The outcome-balanced sampler presents draw labels in one third of training rows, while draw rows
are approximately 53% of the held-out trajectory distribution. Per-class validation confirms the
mechanism: decisive class CE improves strongly, while draw CE changes only `1.10134 -> 1.07666`.

## Frozen sampling correction

Keep total batch size 64 and all compute fixed. Each batch contains:

- 32 rows from the existing outcome- and game-balanced sampler;
- 32 rows from the natural game-balanced replay distribution.

Shuffle the combined indices deterministically. This preserves decisive-class coverage while
restoring the real draw prior. It does not increase positions per step.

## Frozen experiment

- Same decoupled architecture and exact release baseline.
- Same Stage-A auxiliary material qualification.
- Same global-WDL and global+tower-WDL arms.
- Same corrected trajectory-disjoint 148/48 games.
- Same Adam `5e-4`, 400 WDL steps, validation every 20, seeds and hashes.
- Same unchanged WDL gates: micro/macro CE improvement at least 0.10, Brier improvement at least
  0.03, Pearson at least 0.20, adjacent margins at least 0.03, ECE-10 at most 0.12.
- Auxiliary material predictions and release parameters must remain exact.

No threshold, sample size, or run length may change after results. Only a full pass authorizes
continuation action-value ranking.

