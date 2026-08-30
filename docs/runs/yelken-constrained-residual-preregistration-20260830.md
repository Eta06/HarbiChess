# YELKEN constrained plastic-residual preregistration

## Trigger

The frozen-base, 0.1x-base, and mutable-base arms all failed the exact cached
gate. Frozen-base already improved fresh held-out WDL, while old micro CE and
Pearson regressed from the first update. Increasing base plasticity made the
trade-off worse. Fresh generation therefore remains blocked.

## Hypothesis

The remaining failure is destructive interference between historical and fresh
value gradients, not insufficient residual capacity. A Pareto-aware gradient
combiner may find a common descent direction without changing replay exposure,
training duration, policy, search, or the qualified MIHVER base.

## Frozen design

- Use only the frozen-base plastic pathway from the first YELKEN ablation.
- Reuse exactly the same historical/fresh game splits, samplers, seeds, batches,
  learning rate `1e-4`, batch size `1024`, and maximum `40` steps.
- Compute historical and fresh WDL gradients separately on their existing
  512-row halves. Record their dot product, norms, and cosine every step.
- Compare exactly three preregistered combiners:
  1. `mean-control`: arithmetic mean of the two gradients;
  2. `pcgrad`: when gradients conflict, symmetrically remove each gradient's
     component opposing the other, then average;
  3. `mgda`: the analytic minimum-norm convex combination of the two gradients.
- Clip only the final combined gradient to norm `5.0`.
- MIHVER value parameters, release trunk, policy, material head, search, and
  targets remain frozen.

## Gates and selection

Use the exact YELKEN cached gates without tolerance changes: old and fresh micro
CE, macro CE, and Pearson cannot regress from their own baselines; margins,
ECE-10, continuation ranking, and Full Gumbel tactical retention must pass.
Policy and all non-plastic parameters must remain unchanged.

Select the earliest fully passing checkpoint. If more than one combiner passes,
prefer `mgda`, then `pcgrad`, then `mean-control`, because the Pareto solution is
the intended mechanism and the mean is only a control. If none passes, do not
start fresh generation; report that a shared additive residual cannot satisfy
the two distributions under this target/replay regime and return to the value
target or replay-state semantics rather than relaxing the gate.
