# AKTAR Fixed-Shape Inference Preregistration

Date: 2026-08-28  
Status: frozen before benchmark

## Evidence and hypothesis

The selected 0.5 ms wait window did not generalize to the full target workload:
222.80 s versus the 200.61 s variable-batch reference. It nevertheless produced
exact deterministic repeats and zero selected-action/root-visit mismatches on
all 576 rows.

The remaining cost is padding: actual mean batch was 7.64 while every inference
executed a fixed graph shape of 24. A smaller fixed graph may retain deterministic
search while reducing wasted MLX work.

## Frozen benchmark

- Same 48 KOPRU train positions and 64 Full Gumbel simulations as the prior
  wait-window benchmark.
- Fixed shapes: `4`, `8`, `12`, `16`, `24`.
- Wait window: 0.25 ms for every arm.
- 24 search workers; two timed repeats after warm-up.
- Model, seed, ordering, and search configuration unchanged.

## Selection

An arm is eligible only if its repeated selected actions and root visits exactly
match the shape-24 reference and repeated root/soft-target outputs agree within
`1e-12` for that same shape. Choose the eligible arm with minimum median wall
clock. GPU utilization is not a selection metric.

The selected shape must then pass the unchanged full 576-row gates: exact repeat
determinism, zero selected-action/root-visit mismatches against the original
targets, and wall clock no greater than 200.61 s. Failure keeps learner blocked.
