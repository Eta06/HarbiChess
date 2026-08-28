# AKTAR Fixed-Batch Target Result

Date: 2026-08-28  
Artifact: `artifacts/diagnostics/aktar-full-gumbel-targets-20260828-02/result.json`

## Decision

The fixed-shape MLX treatment solved target nondeterminism but failed the frozen
wall-clock gate. Learner training remains blocked.

## Results

- Determinism audit: 32/32 positions passed with exactly `0.0` maximum target
  and root-value delta.
- Selected-action comparison with the original 576 rows: 0 mismatches.
- Serialized root-visit comparison reported 576 mismatches because the loaded
  reference used JSON lists while the in-memory candidate used tuples.
- Rechecking both serialized artifacts showed 0/384 train and 0/192 validation
  root-visit mismatches. This is an artifact comparison bug, not search drift.
- Full run wall clock: 215.05 s versus the frozen 200.61 s reference, a 7.20%
  regression.
- Actual mean queue batch: 7.68 requests despite a fixed padded compute shape
  of 24; padded work therefore dominated the regression.

## Next measurement

Fix only the list/tuple comparison. On a frozen target subset, benchmark batch
wait windows `0.00025`, `0.0005`, `0.001`, and `0.002` seconds with fixed shape
24. Every arm must reproduce selected actions, root visits, root values, and
soft targets within `1e-12`. Select minimum median wall clock; do not select on
GPU utilization. Then rerun the unchanged 576-row provenance workload.
