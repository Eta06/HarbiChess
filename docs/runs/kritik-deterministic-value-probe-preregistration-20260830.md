# KRITIK deterministic value-representation probe preregistration

## Question

Can the current encoder and network learn a deterministic position-value function across held-out
game trajectories? This separates an implementation/representation failure from noisy and scarce
Monte Carlo terminal WDL supervision.

## Frozen dataset and target

- Use the same deduplicated corrected replay pool and trajectory-disjoint 148/48 train/validation
  split as the corrected-replay scale control.
- Select at most 8,192 train and 4,096 validation positions by deterministic round-robin across
  games, preserving broad game and ply coverage.
- Target is the existing deterministic oracle at depth 0: side-to-move material difference passed
  through `tanh(material / 39)`. No game outcome, neural value, or future information is used.
- Optimize mean squared error between the target and WDL expected score `P(win) - P(loss)`.
- Batch size 64, Adam `5e-4`, 200 steps, validation every 20, seed `2026083061`.

## Frozen arms

1. `head-only`: train the existing value conv/hidden/output layers with trunk and policy frozen.
2. `full-representation`: train trunk and value head while keeping the policy head frozen. Policy
   preservation is not claimed; this arm is diagnostic only and its weights cannot become a
   candidate.

## Gates and interpretation

A representation arm qualifies the deterministic probe only if held-out validation simultaneously
has:

- MSE at least 50% lower than the release baseline;
- Pearson correlation at least 0.80;
- mean absolute error at most 0.05.

- If `head-only` passes, the existing trunk already represents simple value features and terminal
  target variance/data scale is the primary blocker.
- If only `full-representation` passes, the encoder/training path works but the policy-trained trunk
  lacks value features; production needs auxiliary value representation learning plus explicit
  policy preservation.
- If neither passes, audit encoder orientation, target sign, gradient/update plumbing, and value
  architecture before any further learning experiment.

This diagnostic cannot authorize arena, continuous learning, generation, or promotion.

