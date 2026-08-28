# ESAS value-gradient interference diagnostic preregistration (2026-08-28)

## Question

The matched control reduced policy cross-entropy but drove validation WDL
cross-entropy from 1.0989 to 1.9210. Is the KOPRU outcome signal itself
unlearnable, or are joint trunk/policy updates destroying value generalization?

## Frozen diagnostic

- Same KOPRU train/validation replay and same release baseline
- Freeze stem, residual trunk, and complete policy head
- Train only the existing WDL value convolution/hidden/output layers
- AdamW `2e-4`, batch 64, 474 steps, seed `2026082869`
- Validate every 79 steps
- Unknown max-ply value rows remain masked
- No auxiliary target, root value, policy loss, new replay, or generation

## Interpretation fixed in advance

- If the best validation WDL cross-entropy improves at least 2% from the
  1.0989 baseline, the data contains learnable value signal and joint-gradient
  interference is the primary next problem.
- Otherwise, the replay split/class balance and outcome volume are the primary
  next problem; do not change the network or add auxiliary heads.

This diagnostic cannot authorize learner/latest, generation, arena, or
promotion. Its result only chooses the next causal branch.
