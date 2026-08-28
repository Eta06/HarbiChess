# CIPA replay-anchored learner result (2026-08-28)

## Decision

CIPA failed its frozen internal holdout gate. External validation, search
qualification, arena, generation, and promotion remain blocked.

## Frozen result

- Fit/holdout: 320/59 positions, split by whole game
- Broad replay anchor: 2,048 fit-side positions
- Anchor weight 0.25: gap closure -16.40%, anchor KL 0.01621, harmful 15.25%
- Anchor weight 1.0: gap closure -4.60%, anchor KL 0.00721, harmful 16.95%
- Anchor weight 4.0: gap closure -1.38%, anchor KL 0.00255, harmful 11.86%

All arms missed the positive verified-gain interval, Spearman, harmful-action,
regret, and gap-closure gates. The strongest anchor constrained off-target
policy drift but did not transfer the SIPER target to unseen games. No
checkpoint was emitted.

## Interpretation

The KANIT failure was not repaired by baseline-policy anchoring. The next audit
must distinguish insufficient learnable signal from contradictory or
position-specific teacher deltas before another learner architecture is tried.

Artifact: `artifacts/diagnostics/cipa-anchored-policy-20260828-01/result.json`
