# MIHENK teacher consistency preregistration

Date: 2026-08-28  
Source replay: `kopru-qualified-replay-20260828-01`  
Source checkpoint: release baseline `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`  
Learner, arena, promotion, and new generation: disabled

## Hypothesis

The depth-1 bootstrap teacher improves independently verified action value in aggregate, but its per-position visit targets may be unstable across clean search budgets. This instability can make soft teacher cross-entropy decrease while teacher top-action agreement and raw tactical ability regress. The audit must separate stable improvement from budget-sensitive or harmful targets before another learner attempt.

## Frozen audit

- Clean search budgets: `64, 128, 256, 512, 800` simulations.
- Teacher: unchanged release policy prior plus deterministic depth-1 tactical/material leaf value.
- Independent verifier: deterministic depth-4 tactical/material continuation value.
- Positions: 96 stratified train records and 48 stratified validation records, selected separately by game-aware replay split.
- Selection seed: `2026082816`.
- Search workers: 24; process oracle workers: 8; MLX batch cap: 48; batch wait: 0.25 ms.
- No root noise, target pruning, continuation/repetition transformation, value-policy adjustment, or architecture change.
- Every row records each budget's policy, top action, top/runner visits, visit share, Q margin, root value, and verified delta versus the raw-network top action.
- Every budget pair records top-action agreement, total variation distance, forward KL, reverse KL, and Jensen-Shannon divergence. KL uses epsilon `1e-12` only to make zero-support comparisons finite and is not a gate by itself.

## Frozen row classes

A row is `stable/high-confidence` only when all conditions hold:

1. All five budgets select the same top action.
2. Maximum pairwise TV among budgets 256, 512, and 800 is at most `0.25`.
3. The minimum normalized visit margin `(leader visits - runner visits) / budget` over 256, 512, and 800 is at least `0.03`.
4. The depth-4 verified action-value improvement over the raw top action is at least `+0.03`.

A row is `harmful` when the 800-simulation top action's independently verified delta versus raw is at most `-0.025`. Every other row is `budget-sensitive/ambiguous`. Harmful takes precedence over stable.

## Frozen consensus gate

The audit is healthy enough to authorize the small ablation only if the held-out validation partition satisfies all conditions:

- stable/high-confidence ratio is at least `20%`;
- harmful ratio is at most `10%`;
- bootstrap 95% lower bound of stable-row verified improvement is strictly positive;
- 512-versus-800 top-action agreement is at least `75%`;
- bootstrap 95% lower bound of the full 800-budget verified improvement is non-negative.

Failure blocks the learner ablation, arena, and generation. Thresholds and sample counts cannot be changed after results are visible.

## Frozen learner ablation

If and only if the consensus gate passes, run three arms from the same release weights with identical game-balanced sample order:

1. `raw-control`: raw-network policy target on every audited training row.
2. `all-teacher`: stored clean 64-simulation teacher target on every row.
3. `consensus-gated`: stored teacher target only for stable/high-confidence rows; raw-network target for ambiguous or harmful rows.

Records are not removed. Position coverage, value labels, and batch composition remain identical; only the policy target differs. All arms use policy-only AdamW, learning rate `2e-4`, weight decay zero, batch 64, exactly 240 steps, seed `2026082817`, no hard-top auxiliary, and no continuation adjustment.

The consensus-gated arm passes transfer only if, on the separately audited validation records, it:

- beats `all-teacher` teacher top-action agreement by at least `3` percentage points on stable rows;
- does not worsen stable-row legal teacher cross-entropy versus `all-teacher`;
- preserves raw-policy agreement on ambiguous/harmful rows at least as well as `all-teacher`;
- does not regress release raw tactical or 64-simulation tactical solve count;
- has finite losses and gradients within the existing gradient limit.

Passing this small diagnostic does not authorize arena or generation. It only authorizes a fresh, separately preregistered replay-scale confirmation.

## Performance boundary

Performance experiments use the release baseline and a separate frozen benchmark. No performance code change may enter while the quality audit is running. A performance change is retained only with lower end-to-end wall time, higher simulations/s or games/hour, and bitwise-identical selected actions, visits, Q values, and root values on the equivalence suite.
