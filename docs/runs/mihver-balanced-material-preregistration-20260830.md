# MIHVER balanced material objective preregistration

## Measured mechanism

Distributional material supervision fixed draw calibration but failed the unchanged scalar gate.
On the frozen first batch, the soft-WDL cross-entropy gradient norm is `10.40779`, expected-score
MSE gradient norm is `0.266216`, their ratio is `39.095`, and cosine is `0.0383`.

The distributional loss therefore overwhelms the nearly orthogonal scalar-value signal.

## Frozen objective

Use:

`soft WDL cross-entropy + 39 * expected-score MSE`

The coefficient is fixed from the measured gradient ratio before results. No weight sweep or
post-result threshold change is allowed.

## Frozen experiment and gates

- Same count-scaled invariant network and bitwise-preserving initialization.
- Same global-linear and invariant-tower arms.
- Same trajectory-disjoint 8,192/4,096 material positions.
- Same Adam `2e-3`, batch 64, 200 steps, validation every 20, seed `2026083061`.
- Same hard material gates: MSE reduction at least 50%, Pearson at least 0.80, MAE at most 0.05.
- Record soft-target CE and draw-probability MAE without substituting them for the hard gates.
- Same frozen parameter hashes and model-selection rule.

A pass authorizes rerunning the already frozen WDL calibration design from the new qualified model.
Failure ends scalar loss-weight experiments and requires separately parameterized value and draw
outputs. Continuous learning, generation, arena, and promotion remain blocked.

