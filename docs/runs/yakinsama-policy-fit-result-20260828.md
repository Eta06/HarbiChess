# YAKINSAMA low-rank policy convergence result (2026-08-28)

## Decision

The train-only convergence gate passed first at step 960. That checkpoint is
frozen for one fresh validation. It does not yet authorize search
qualification, arena, generation, or promotion.

## Evidence

Step 480 failed only the harmful gate at 11.61%. Step 960 closed 62.41% of the
reducible KL gap, reached 0.6578 teacher Spearman, produced verified gain with
a +0.0338 to +0.0649 95% interval, reduced harmful selection to 8.97%, held
mean regret at 0.0663, and covered a verified-best action in the top 16 on
90.24% of rows. Step 1920 also passed, but the preregistered earliest-passing
rule selects step 960.

## Frozen artifact

- `artifacts/diagnostics/yakinsama-policy-20260828-01/result.json`
- Candidate SHA-256: `c114b9cffce0f12c993f572de2641fedab012a027f4792bea3030b9054c33e2a`
