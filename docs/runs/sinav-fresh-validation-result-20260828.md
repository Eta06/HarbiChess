# SINAV fresh validation result (2026-08-28)

## Decision

The fresh teacher failed before candidate evaluation. The frozen YAKINSAMA
step-960 candidate was not loaded or scored on this set. Search qualification,
arena, generation, and promotion remain blocked.

## Evidence

On 95 labelable fresh validation positions, stable-Q/verifier Spearman was
0.3152, below the frozen 0.35 threshold. Labelable coverage was 98.96%, stable
visit mass 88.70%, conservative verified-gain 95% interval +0.0296 to +0.0730,
harmful ratio 7.37%, and mean regret 0.0381. The raw dataset also failed with
800-Q/verifier Spearman 0.3365 and top-two cross-budget overlap 73.44%.

This set is now diagnostic-only and cannot later become promotion evidence.
The next step is to segment its teacher instability, preregister a search-side
hypothesis, and test that hypothesis on another fresh set before resuming
learner validation.

## Frozen artifacts

- `artifacts/diagnostics/sinav-raw-action-value-dataset-20260828-01/dataset.json`
- `artifacts/diagnostics/sinav-uncertainty-labels-20260828-01/labels.json`
