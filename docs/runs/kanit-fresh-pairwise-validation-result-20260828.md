# KANIT fresh pairwise validation result (2026-08-28)

## Decision

The fresh MARJ teacher and SIPER target passed, but the frozen YAKINSAMA
candidate failed transfer. Search qualification, arena, generation, and
promotion remain blocked.

## Teacher and target

Decisive-pair concordance was 0.7294 with a 95% interval of 0.6785 to 0.7814
across 7,985 pairs and 82.11% informative-position coverage. Conservative
teacher gain, harm, regret, support, and stable-mass gates passed. The SIPER
target had verified expected-gain interval +0.00780 to +0.01421, 4.21% harmful
top actions, zero harmful expected-value rows, and 90.60% effective-action
retention.

## Candidate

The step-960 candidate produced -37.60% reducible-gap closure on fresh data,
0.1408 teacher Spearman, verified-gain interval -0.0187 to +0.0458, 13.68%
harmful actions, 0.1156 regret, and 78.95% top-16 coverage. Raw tactical solves
regressed from 1 to 0; WDL logits remained bitwise identical.

The candidate overfit the sparse high-budget train set. The next learner
hypothesis must mix qualified targets with broad replay policy anchoring and
prove internal holdout transfer before another external validation.

## Frozen artifacts

- `artifacts/diagnostics/kanit-pairwise-teacher-20260828-01/result.json`
- `artifacts/diagnostics/kanit-policy-target-20260828-01/targets.json`
- `artifacts/diagnostics/kanit-candidate-validation-20260828-01/result.json`
