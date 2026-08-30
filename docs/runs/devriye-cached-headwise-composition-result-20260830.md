# DEVRIYE cached head-wise composition result

## Decision

The head-wise checkpoint mechanism passed every update-level guardrail on the
cached `devriye-continuous-pilot-20260830-12` update-1 inputs. This validates the
mechanism but does not authorize generation because the evidence reuses an
observed replay/teacher set. The preregistered fresh three-update run remains
mandatory.

## Selected checkpoints

- Policy: local step 20.
- Invariant/global WDL: local step 1.
- Shared trunk and auxiliary material output: bitwise unchanged.
- Adam moments: reset after composition; weights and learner-step lineage kept.

## Joint result

| Metric | MIHVER start | Composed candidate |
|---|---:|---:|
| Policy validation CE | 2.920710 | **2.902050** |
| Policy top-action agreement | 8.85% | **21.35%** |
| WDL micro CE | **0.912848** | 0.915287 |
| WDL macro CE | 0.937718 | **0.936241** |
| WDL expected-score Pearson | 0.452612 | 0.452388 |
| Continuation Spearman | 0.073161 initial reference | **0.215836** |
| Continuation top agreement | 37.50% initial reference | **46.88%** |
| Full Gumbel tactical | 5/8 | **6/8** |

The unchanged 4-pair/32-simulation mini arena scored `2-6-0`, expected score
0.625, with no threefold termination. Every fixed policy, WDL, continuation,
tactical, material, and catastrophic-arena threshold passed.

## Next action

Run the full fresh three-update chain with seed `2026083901`. No cached metric
may be used to rescue a failure in that run.
