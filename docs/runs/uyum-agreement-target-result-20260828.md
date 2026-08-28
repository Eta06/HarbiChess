# UYUM common-direction target result

Date: 2026-08-28  
Source commit: `21dee7d`  
Artifact: `artifacts/diagnostics/uyum-agreement-target-20260828-01/agreement.json`  
Decision: teacher gate failed; learner, arena, generation, and promotion remain blocked

## Result

UYUM projected the raw policy only along probability directions independently supported by both clean 512- and 800-simulation searches. It used the exact frozen DENGE positions, verifier, thresholds, and bootstrap design. No parameter changed after results became visible.

| Metric | Train | Validation | Gate |
|---|---:|---:|---:|
| Qualified rows | 20/96 (20.83%) | 8/48 (16.67%) | at least 20% |
| Harmful rows | 0/96 | 0/48 | at most 10% |
| Mean expected-value delta vs raw | +0.04564 | +0.05826 | positive 95% lower bound |
| Expected-value delta 95% interval | [+0.02615, +0.06940] | [+0.02446, +0.09973] | passed |
| Qualified delta 95% interval | [+0.06578, +0.21867] | [+0.05706, +0.41491] | passed |
| Mean anchor-to-target TV | 0.10186 | 0.09379 | at most 0.125 |
| Mean delta vs 800 policy | -0.00749 | -0.00607 | no worse than -0.01 |

The validation qualified ratio again failed at 16.67%, so the conditional learner ablation was not run. The result is not rescued by the passing train ratio.

## Diagnosis

Two materially different uncertainty-preserving constructions qualified the same number of validation rows. Common-direction filtering did not reduce harmful mass or raise per-row expected improvement enough to change coverage. The dominant failures remained expected-value improvement below +0.02 (28 rows) and harmful mass above 10% (15 rows).

This rejects another visit-probability mixing iteration. The next diagnostic must determine whether complete child-Q estimates from 512/800 search rank independently verified actions better than visit counts. If Q is reliable while visit policy is diffuse, the learner target should represent an explicit policy-improvement operator over Q. If Q is also unreliable, the blocker remains search allocation/value estimation and learner experiments stay closed.
