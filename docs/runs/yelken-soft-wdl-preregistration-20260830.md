# YELKEN uncertainty-preserving WDL target preregistration

## Evidence

The replay target-conflict audit found 101 exact encoded states shared by the
historical and fresh pools. Fifty-one of them carry different one-game WDL
outcomes across pools, with mean merged target entropy `1.0186` bits. The
two-gradient ablation also observed cosine values down to approximately `-0.95`;
neither PCGrad nor two-objective MGDA passed the unchanged gates.

## Hypothesis

Repeated identical states should not be forced toward mutually exclusive one-hot
Monte Carlo outcomes. Merging their empirical outcome observations into a soft
WDL target preserves uncertainty and removes an avoidable source of destructive
gradient interference. A generalized Pareto combiner must also represent old
micro CE, old macro CE, old Pearson, fresh micro CE, fresh macro CE, and fresh
Pearson separately instead of reducing each domain to one CE gradient.

## Frozen ablation

- Keep MIHVER fully frozen and train only the zero-output plastic residual.
- Reuse the exact YELKEN historical/fresh splits, seeds, learning rate, maximum
  40 steps, and no-new-generation rule.
- Aggregate outcomes only for byte-identical encoded states across the combined
  fit pools. Every occurrence of such a state receives the same empirical
  three-class probability target. Unique states retain one-hot WDL targets.
- Validation labels remain untouched and game-disjoint; soft targets are built
  from fit rows only.
- Compare three fixed arms:
  1. `onehot-generalized-mgda`: six-objective MGDA without state aggregation;
  2. `soft-state-mean`: aggregated targets with an equal mean gradient;
  3. `soft-state-generalized-mgda`: aggregated targets with six-objective MGDA.
- Generalized MGDA first normalizes each objective gradient to unit L2 so a
  numerically small loss cannot win only by scale, then uses a deterministic
  100-iteration Frank-Wolfe minimum-norm convex solver. Report objective weights,
  gradient Gram/cosine evidence, and target aggregation counts.
- Policy, search, replay exposure, MIHVER, and validation gates do not change.

## Gate

Use the existing exact old/fresh micro CE, macro CE, Pearson, margin, ECE,
continuation, tactical, immutable-parameter, and policy-preservation gates.
Select the earliest fully passing checkpoint; arm preference is
`soft-state-generalized-mgda`, then `soft-state-mean`, then
`onehot-generalized-mgda`.

If no arm passes, do not run a fresh pilot. That result rejects both exact-state
softening and multiobjective gradient geometry as sufficient fixes; the next
decision must revisit Monte Carlo value-target scale/freshness or the exact
cross-domain gate assumption, with new preregistration rather than post-result
threshold changes.
