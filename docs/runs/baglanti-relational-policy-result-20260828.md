# BAGLANTI relational policy representation result (2026-08-28)

## Decision

The relational adapter failed train-fit qualification. Fresh validation,
search qualification, arena, generation, and promotion remain blocked.

## Evidence

After 480 frozen steps, the adapter closed 6.28% of the reducible KL gap,
reached 0.2565 teacher Spearman, and selected harmful actions on 12.14% of train
rows. Its verified-gain interval was positive and WDL remained bitwise stable,
but three mandatory gates failed.

Both compact structured adapters underfit, while the rank-32 low-rank global
adapter previously closed 46.31% of the same target gap at 480 steps. The next
controlled diagnostic therefore measures low-rank convergence at fixed
480/960/1920 checkpoints. This is a measured learning-curve test, not an
automatic authorization to increase normal training.

## Frozen artifact

- `artifacts/diagnostics/baglanti-relational-policy-20260828-01/result.json`
