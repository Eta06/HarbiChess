# AKTAR Target Provenance Result

Date: 2026-08-28  
Artifact: `artifacts/diagnostics/aktar-full-gumbel-targets-20260828-01/result.json`

## Decision

The first Full Gumbel target artifact failed the frozen `1e-12` determinism
gate. Learner training did not start. Thresholds and target rows remain frozen.

## Data quality

- Train targets: 384 positions from 57 games.
- Validation targets: 192 positions from 23 games.
- All 24 preregistered phase/tactical/outcome composite strata are represented
  equally: 16 train and 8 validation rows each.
- Train mean raw-to-teacher TV: 0.72974; KL: 1.85508; argmax change: 86.46%.
- Validation mean raw-to-teacher TV: 0.73886; KL: 1.84330; argmax change: 91.15%.
- Model, shard-file, and shard-payload hashes matched provenance.

## Determinism failure

The original 32-row audit compared the parallel generation path with a serial
repeat and failed. A causal rerun measured:

- Serial batch-1 repeat: 3 selected-action mismatches, 9 root-visit
  mismatches, maximum target delta 0.806648, maximum root-value delta
  0.000081427.
- Repeated 24-worker variable-batch path: 0 selected-action mismatches, 0
  root-visit mismatches, maximum target delta 0.000404204, maximum root-value
  delta 0.000000157.

The failure is therefore not search RNG. MLX evaluations at different batch
shapes differ slightly; Full Gumbel's close completed-Q comparisons can amplify
those differences into a different search path.

## Remediation boundary

Do not relax the tolerance or train from this artifact. Add an opt-in fixed-size
padded MLX inference batch, benchmark it against the existing variable batching,
and require:

1. repeated fixed-batch target/action/visit/root outputs within `1e-12`;
2. identical selected action and root visits against the original variable-batch
   targets on all 576 frozen rows;
3. no wall-clock regression versus the variable-batch target run.

Only a fresh artifact passing those conditions may authorize the already
preregistered learner-transfer experiment.
