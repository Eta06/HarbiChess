# DENGE constrained calibration preregistration

Date: 2026-09-01

## Prior result

DENGE-1 failed and remains failed. Its fresh-only scalar improved fresh test
ECE-10 by 0.096483, CE by 0.055729, Brier by 0.028634, and Pearson by 0.006537.
It preserved policy logits bitwise, improved Full Gumbel tactical solve rate from
6/8 to 7/8, and kept continuation ranking inside both preregistered margins.
However, old-distribution Pearson deteriorated by 0.013449 and exceeded the
fixed 0.010 non-inferiority margin.

The next hypothesis is not a relaxed gate. Fresh-only temperature selection can
over-sharpen the stable distribution even while all proper scoring rules
improve. DENGE-2 retains the same one-scalar family but constrains scale selection
with separate historical guard data.

## Frozen DENGE-2 diagnostic

- Source network, replay, search, fit/test fresh partition, and all DENGE-1
  capability gates remain unchanged.
- The old qualification games are deterministically divided into game-disjoint
  guard and test halves before scale selection.
- First compute the fresh-fit CE-optimal positive scalar.
- Move from identity scale toward that optimum only as far as the old guard
  expected-score Pearson permits.
- The local guard margin is fixed at 0.005 deterioration, stricter than the
  unchanged final old-test margin of 0.010.
- The old test half is never used for scalar selection.
- No policy, representation, class bias, loss weight, exposure, replay, search,
  or learner parameter changes.

DENGE-2 passes only if the original DENGE-1 test gates all pass:

1. Fresh test ECE-10 improves by at least 0.020; CE and Brier do not regress;
   Pearson deterioration is at most 0.005.
2. Old test CE and Brier deterioration is at most 0.003, Pearson deterioration
   is at most 0.010, and absolute ECE-10 is at most 0.120.
3. Policy logits remain bitwise equal.
4. Full Gumbel loses no solved case and solves at least 5/8.
5. Continuation mean Spearman deterioration is at most 0.020 and verified-top
   agreement loses at most one of 1,440 positions.

The margins will not be changed after observing DENGE-2. This remains a
diagnostic-only use of already inspected PUSULA-16 data. Passing authorizes a
fresh cumulative pilot, not promotion or production training.
