# PUSULA powered old-calibration preregistration

## PUSULA-11 verdict

`pusula-continuous-pilot-20260831-11` remains failed. It is not reclassified.
All three updates, exact resume, policy imitation, material, Full Gumbel
tactical retention, final search arena, fresh CE/macro CE/Brier/Pearson/ECE,
powered continuation, and four of six historical checks passed. Production
remains disabled because historical CE's bootstrap upper bound was `0.003116`
against `0.003`, and historical ECE non-inferiority failed.

The result establishes that independent-game scale fixed learner transfer:
updates had 273, 275, and 273 known terminal games, while the final fresh CE
improvement interval was `[0.003907, 0.005199]`. The remaining blocker is
old-function calibration retention and its measurement, not teacher quality or
fresh data absorption.

## Old ECE power correction

The old gate still used only 48 historical games. Its ECE bootstrap interval
width implies an approximate game-cluster standard deviation of `0.070`. For a
one-sided `0.010` non-inferiority boundary, assumed zero deterioration, alpha
`0.05`, power `0.80`, and 15% inflation, the normal approximation requires about
350 known games. This metric was omitted from the earlier CE-only power plan.

PUSULA-12 therefore generates a sealed old-distribution qualification set with
1,536 phase-balanced attempts from untouched historical validation states using
the immutable update-0 MIHVER network. At least 384 must reach a known terminal
result. Max-ply games remain unknown and cannot become draw labels. Neither its
records nor labels participate in training or checkpoint selection. The
original 48 historical games remain a separate point-margin compatibility gate;
they are not asked to provide the powered confidence interval.

## Frozen learning change

PUSULA-11 used equal aggregate weight for 32 historical MIHVER-distillation and
32 fresh terminal-outcome examples. PUSULA-12 keeps the exact batch composition,
40 steps, learning rate, architecture, and optimizer, but fixes historical
distillation sample weight at `2.0` and fresh weight at `1.0`, normalized by
total weight. This directly penalizes plastic residual drift instead of shrinking
a completed checkpoint after seeing its result.

The new run is `pusula-continuous-pilot-20260831-12`, seed `2026091101`. It keeps
768 attempts and minimum 192 known games per update; all policy/search settings,
residual alpha candidates, local old/fresh gates, 1,440-position continuation,
64-game arena, 2,688-attempt fresh qualification, 744-known floor, cumulative
margins, and 20,000 bootstrap samples are unchanged.

Final historical confidence intervals use only the sealed MIHVER-generated old
qualification set. The original historical holdout must independently remain
inside the same point margins. The run fails on either old gate, any fresh gate,
continuation, tactical/search, data-integrity, or resume failure. Passing can
authorize production continuous-loop integration, never automatic release
promotion.
