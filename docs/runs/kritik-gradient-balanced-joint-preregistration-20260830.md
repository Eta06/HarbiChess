# KRITIK gradient-balanced joint-transfer preregistration

## Measured mechanism

At the frozen release baseline, using the first deterministic policy and value batches from the
shared-representation audit, the policy gradient norm on the shared trunk is `0.1056409`; the WDL
gradient norm is `0.0244421`. The policy-to-value norm ratio is `4.3221` and their cosine is
`0.0668`. Equal scalar loss weights therefore let the small Full Gumbel imitation set dominate
shared representation updates while providing almost no aligned value signal.

This experiment tests that measured mechanism. It is not a post-result threshold change.

## Frozen change

- Policy loss weight: `0.25`.
- WDL loss weight: `1.0`.

The `0.25` value is fixed from the measured `1 / 4.3221` scale, rounded conservatively. No dynamic
gradient manipulation is introduced.

Everything else remains identical to
`kritik-shared-representation-preregistration-20260830.md`: release baseline, qualified Full
Gumbel targets, schema-12 corrected replay, separate batches of 64, AdamW `1e-4`, weight decay
`1e-4`, 400-step maximum, 20-step validation, patience 6, seeds, architecture, continuation
ranking, tactical suite, and all numeric gates.

## Decision rule

- All WDL, policy-imitation, calibration, continuation-ranking, and Full Gumbel tactical gates
  must pass before arena.
- Policy imitation alone is failure.
- A WDL improvement that loses Full Gumbel policy/tactical ability is failure.
- If value remains collapsed, do not change weights again or lengthen training. Move to a
  deterministic value-representation probe and then audit target variance/auxiliary value
  supervision.

Continuous learning, generation, arena, and promotion remain blocked unless their existing
prerequisites are reached.

