# VERI teacher-coverage transfer result (2026-08-28)

## Decision

VERI failed. Search qualification, arena, generation, and promotion remain
blocked. Increasing fresh labelled coverage alone did not solve transfer.

## Evidence

The expanded uncertainty gate passed on 379/384 train and 95/96 validation
positions. Validation stable-Q/verifier Spearman was 0.3520, conservative
verified-gain 95% interval was +0.0603 to +0.1381, harmful ratio was 6.32%, and
mean regret was 0.0434. Six positions with no action under the frozen drift
cutoff were explicitly quarantined; labelable coverage remained above 98%.

With unchanged learner compute, no checkpoint passed. Train teacher Spearman
reached only 0.2910 and validation Spearman fell from 0.1024 to 0.0401. The best
validation verified-gain lower bound occurred at step 240, but cross entropy,
rank correlation, regret, top-16 coverage, and tactical retention still failed.

The audit exposed a target-semantics error in the attempted direct transfer:
the uncertainty label `weight` is a confidence/loss weight for a Q observation,
not a desired policy probability. Normalizing it as policy mass can reward an
action merely because its Q estimate is stable, even when that Q is low. The
next target must keep confidence separate and derive policy improvement from
the Q values themselves.

## Frozen artifacts

- `artifacts/diagnostics/veri-raw-action-value-dataset-20260828-01/dataset.json`
- `artifacts/diagnostics/veri-uncertainty-labels-20260828-01/labels.json`
- `artifacts/runs/veri-uncertainty-policy-20260828-01/result.json`
