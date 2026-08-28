# DENGE uncertainty-preserving consensus target result

Date: 2026-08-28  
Source commit: `6c62f46`  
Artifact: `artifacts/diagnostics/denge-consensus-target-20260828-01/consensus.json`  
Decision: teacher gate failed; learner, arena, generation, and promotion remain blocked

## Frozen execution

The audit reused the exact 96 train and 48 validation positions and clean 256/512/800 visit distributions from MIHENK. Every legal action was independently evaluated with the deterministic depth-4 tactical/material verifier, so expected values cover the complete policy distribution rather than a truncated top-k subset. The run completed in 20.88 seconds.

The DENGE policy was the preregistered equal arithmetic mixture of the 512- and 800-simulation clean visit distributions. No temperature, pruning, sharpening, or threshold changed after results became visible.

## Results

| Metric | Train | Validation | Gate |
|---|---:|---:|---:|
| Qualified rows | 21/96 (21.88%) | 8/48 (16.67%) | at least 20% |
| Harmful rows | 0/96 | 0/48 | at most 10% |
| Mean expected-value delta vs raw | +0.04879 | +0.06091 | positive 95% lower bound |
| Expected-value delta 95% interval | [+0.02851, +0.07588] | [+0.02774, +0.10364] | passed |
| Qualified delta 95% interval | [+0.06844, +0.21747] | [+0.06137, +0.39763] | passed |
| Mean anchor-to-consensus TV | 0.11181 | 0.09959 | at most 0.125 |
| Mean delta vs 800 policy | -0.00434 | -0.00341 | no worse than -0.01 |
| Mean harmful probability mass | 9.59% | 7.87% | descriptive |

The single failed gate was validation target coverage: 16.67% instead of the frozen 20% minimum. The learner ablation was therefore not run. Treating the result as a pass because it missed by only two rows would be post-hoc threshold relaxation.

The arithmetic consensus is nevertheless informative. It is stronger than the raw distribution with a strictly positive interval, produces no harmful validation rows, converges from the 256/512 anchor, and loses only 0.0034 expected value relative to the 800 distribution. This supports uncertainty preservation while rejecting unconditional averaging as the final target.

## Failure anatomy and next hypothesis

Validation disqualification counts overlap:

- expected-value improvement below +0.02: 26 rows;
- harmful probability mass above 10%: 15 rows;
- anchor-to-consensus TV above 0.20: 6 rows;
- no overlap in the two top-two sets: 3 rows.

The mean target is good, but simple averaging retains probability changes that only one high-budget search supports. The next frozen target should preserve multiple actions while moving probability away from raw only where both 512 and 800 independently agree on the direction of the policy improvement. This is a target-representation correction, not a relaxation of DENGE's qualification thresholds.
