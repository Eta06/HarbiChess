# HACIM expanded high-budget label result (2026-08-28)

## Decision

HACIM failed both its frozen raw-Q gate and its separate uncertainty-label
gate. No policy target was built; learner, external qualification, arena,
generation, and promotion remain blocked.

## Scale and runtime

- 2,048 fresh train positions and 256 fresh validation positions
- Every prior TERAZI/DOKU/OLCEK/BAG/VERI/SINAV/KANIT identity excluded
- 512 and 800 clean searches plus depth-4 verification of every legal action
- Wall time: 3,563.82 seconds
- MLX positions: 2,943,712 (826.0 positions/s end to end)
- Backend time: 2,376.58 seconds; 615,594 batches; mean batch 4.78, max 16

## Raw-Q evidence

| Metric | Train | Validation | Gate |
| --- | ---: | ---: | ---: |
| Q/verifier Spearman | 0.3281 | 0.3191 | at least 0.35 |
| Cross-budget Q Spearman | 0.7749 | 0.7650 | at least 0.70 |
| Mean Q drift | 0.01544 | 0.01556 | at most 0.03 |
| Top-two overlap | 70.90% | 70.51% | at least 75% |
| Top-Q verified gain 95% interval | [+0.0580,+0.0732] | [+0.0559,+0.0970] | positive |
| Harmful top-Q | 7.18% | 7.03% | at most 10% |
| Mean top-Q regret | 0.0509 | 0.0458 | at most 0.10 |

The raw gate failed Q/verifier rank correlation and top-two overlap.

## Uncertainty-label evidence

Action-level drift filtering retained 2,039/2,048 train and 254/256 validation
rows. Common support, stable visit mass, conservative-action strength, harm,
and regret all passed. Stable-Q/verifier Spearman was 0.3094 train and 0.3096
validation, below the unchanged 0.35 gate. The uncertainty gate therefore
failed its only remaining reason.

## Interpretation

The earlier 384/96 VERI pass was a small-sample overestimate around a marginal
threshold. HACIM provides substantially tighter evidence: depth-1 leaf values
produce a strong aggregate top action, but the full Q surface is not aligned
well enough with the independent depth-4 verifier to supervise a soft policy
distribution. More learner capacity, epochs, or replay anchoring cannot repair
an unqualified teacher distribution.

The next causal experiment should stay on search/value: on a frozen diagnostic
subset, replace only depth-1 leaf evaluation with a deeper short-horizon oracle
and measure Q/verifier rank alignment, top-action strength, cross-budget
stability, and wall time. A learner should reopen only if that teacher passes
on an independently fresh set.

Artifacts:

- `artifacts/diagnostics/hacim-raw-action-value-dataset-20260828-01/dataset.json`
- `artifacts/diagnostics/hacim-uncertainty-labels-20260828-01/labels.json`
