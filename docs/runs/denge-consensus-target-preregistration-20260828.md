# DENGE uncertainty-preserving consensus target preregistration

Date: 2026-08-28  
Parent evidence: `artifacts/diagnostics/mihenk-teacher-consistency-20260828-01/consistency.json`  
Baseline: the unchanged qualified KOPRU release checkpoint  
Decision scope: teacher-target qualification first; learner only after a pass; no arena, generation, or promotion

## Hypothesis

MIHENK showed that 800-simulation clean search improves independently verified action value in aggregate, while 512/800 argmax agreement is only 66.67%. The failure may be representational: cross-entropy against either visit distribution penalizes the learner for preserving probability on a near-tied alternative selected by the other budget.

The frozen DENGE target is the equal arithmetic mixture of the independently clean 512- and 800-simulation visit distributions. It preserves all legal support and uncertainty; it does not prune, sharpen, temperature-scale, or replace the target with its argmax. The comparison anchor is the equal 256/512 mixture, which tests whether the soft target converges as compute increases instead of merely benefiting from the algebraic fact that a mixture is halfway between its two inputs.

## Frozen teacher audit

Use exactly the 96 train and 48 validation positions selected by MIHENK. Reuse its clean visit distributions and independently evaluate every legal action with the same deterministic depth-4 tactical/material verifier. This permits exact expected verified value for the full raw-network, 512-search, 800-search, and DENGE distributions; no top-k truncation is allowed.

For each position report:

- TV, symmetric KL/Jensen-Shannon divergence, and top-2 mass overlap between the 256/512 anchor and 512/800 consensus;
- entropy and effective action count for raw, 512, 800, and consensus policies;
- full-policy verified expected value and delta versus raw policy;
- probability mass on verified improvements of at least `+0.03` and verified harms of at most `-0.025`, relative to the raw-policy argmax;
- consensus top-action verified delta and the 512/800 top-action/value relationship.

A row is learner-qualified only when all conditions hold:

1. anchor-to-consensus TV is at most `0.20`;
2. top-2 action-set overlap is non-empty;
3. consensus expected verified value exceeds raw-policy expected value by at least `+0.02`;
4. harmful probability mass is at most `0.10`.

The validation teacher gate passes only when:

- at least `20%` of rows qualify;
- at most `10%` of rows have consensus expected-value delta at or below `-0.025`;
- the bootstrap 95% lower bound of full-validation consensus expected-value improvement is strictly positive;
- the bootstrap 95% lower bound of qualified-row improvement is strictly positive;
- mean anchor-to-consensus TV is no greater than `0.125`;
- mean consensus verified expected value is not below mean 800-policy value by more than `0.01`.

Bootstrap uses 2,000 resamples and seed `2026082818`. These thresholds and sample counts will not change after the result is visible.

## Conditional fixed-compute learner ablation

Run this section only if the validation teacher gate passes. Keep the release architecture, baseline weights, train/validation positions, game-balanced sampler, batch 64, policy-only AdamW, learning rate `2e-4`, zero weight decay, exactly 240 steps, checkpoint schedule, tactical suite, and seeds identical between arms.

Three arms are frozen:

1. `raw-control`: raw-network policy on every selected row;
2. `single-800`: clean 800-search policy on learner-qualified rows and raw policy elsewhere;
3. `consensus`: DENGE 512/800 soft target on learner-qualified rows and raw policy elsewhere.

All 96 train rows and all 48 validation rows remain present in every arm. Value targets and value loss are disabled in this target-transfer ablation, while the unchanged broader KOPRU validation replay is used to measure WDL/calibration retention.

The consensus arm passes transfer only if one preregistered checkpoint:

- lowers qualified-row consensus legal cross-entropy by at least `2%` from baseline;
- beats `single-800` qualified-row consensus cross-entropy;
- does not reduce qualified-row consensus probability-mass capture;
- preserves the baseline raw and 64-search tactical solve counts;
- keeps broader-replay WDL cross-entropy within `1.02x` baseline and expected-score ECE within `+0.02`;
- has finite losses/gradients with clipped gradient norm at most `5.0`.

Checkpoint selection is by the complete gate above, then lowest qualified-row consensus cross-entropy. Validation loss alone cannot authorize a candidate. Even a passing ablation authorizes only a subsequent fresh teacher/replay design decision; arena, generation, and promotion remain disabled.
