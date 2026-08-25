# KASIM value-aware repetition ablation — 2026-08-26

## Decision

The corrected v6 candidate passed paired strength, win-rate, and decisive-performance
guardrails against continuation-off, but failed the avoidable-threefold guardrail by a
wide margin. It was not promoted and no new generation was started. The champion and
the retained v4/v5 evidence remain unchanged.

## V5 branch and paired-impact audit

V5 had removed six v4 branches solely because at least one short rollout looped. Five
of those six roots were equal or defensive according to the champion (`root_value`
between -0.021 and +0.021). Only one had a clear positive root value (+0.092).

Across the exact 200 paired V4/V5 arena assignments, V5 did not merely convert
avoidable draws into wins:

| V4 → V5 result | Games |
| --- | ---: |
| Loss → draw / win | 24 / 1 |
| Draw → loss / win | 27 / 14 |
| Win → draw / loss | 9 / 2 |
| Unchanged | 123 |

V4 and V5 both scored 43.75%. Avoidable-threefold changed in both directions: 43
V4-avoidable games became non-avoidable, while 35 previously non-avoidable games
became avoidable. The net improvement was eight games, from 148 to 140. This showed
that probability-only deletion was directionally useful but too noisy to distinguish
legitimate defensive draws from squandered advantages.

Audit artifact: `artifacts/audits/kasim-v5-paired-impact-20260826-01/impact.json`.

## Value-aware target

Target schema v6 preserves schema 3/4/5 reader compatibility and stores, per branch:

- short-horizon repetition probability and Wilson upper bound;
- loop-value sample count and exact-value sample count;
- mean and lower-bound loop value;
- risk-adjusted branch lower value;
- evaluated root value and the frozen advantaged-root threshold.

The experiment froze `root_value > +0.05` as the advantaged context before generating
targets. Roots at or below the threshold retained their original arena/MCTS policy,
including legitimate repetition mass. Advantaged roots used

`(1 - observed risk) × branch LCB + observed risk × min(branch LCB, loop-value LCB)`.

Compute remained fixed at 16 rollouts, three plies, and 32 MCTS simulations. Training
remained 200 attempted steps, batch size 64, seed `2026082504`, 55 continuation
records, and 25% continuation exposure.

## Exact draw correction

The first KASIM audit treated a single claimable-threefold value as statistically
uncertain and assigned a -1 lower bound. This was incorrect: accepting a claimable
draw has exact value 0. The preliminary `01` run was retained for auditability but was
not used for the final decision.

Schema and generator were corrected to mark exact loop-value samples. Re-running the
same deterministic audit showed all seven loop events were exact claimable draws.
The final v6 shard preserved original policies at 52 equal/defensive roots and applied
value-aware redirects at all three roots above +0.05.

| Corrected audit metric | Result |
| --- | ---: |
| Roots / branches | 55 / 62 |
| Exact claimable-draw loop events | 7 |
| Original defensive policies | 52 |
| Value-aware redirects | 3 |
| Audit time | 52.96 s |
| Neural positions / average batch | 91,405 / 29.09 |

## Fixed-compute training

The corrected candidate stopped because it reached the fixed 200-step limit, not
because of early stopping, and restored the best-validation checkpoint at step 190.

| Metric | Corrected v6 |
| --- | ---: |
| Training time | 1.87 s |
| Initial / final train loss | 9.54887 / 5.56799 |
| Initial / best validation loss | 9.54764 / 7.48944 |
| Continuation records / exposure | 55 / 25% |

Validation loss was worse than v4/v5, but the candidate proceeded to the frozen arena
because validation was not the promotion criterion.

## Corrected 200-game arena

| Arm | W-D-L | Score | Avoidable threefold | Decisive score |
| --- | ---: | ---: | ---: | ---: |
| Continuation off | 2-155-43 | 39.75% | 144 / 200 | 4.44% |
| V4 confidence-gated | 11-153-36 | 43.75% | 148 / 200 | 23.40% |
| V5 probability-risk | 15-145-40 | 43.75% | 140 / 200 | 27.27% |
| Corrected v6 value-aware | 10-157-33 | **44.25%** | **151 / 200** | 23.26% |

### Frozen gate versus continuation off

| Guardrail | Estimate | Confidence interval | Result |
| --- | ---: | ---: | --- |
| Paired score difference | +4.50 pp | two-sided 95% +0.50 to +8.75 pp | **Pass** |
| Avoidable-threefold difference | +3.50 pp | one-sided 95% -4.00 to +11.00 pp | **Fail** |
| Win-rate difference | +4.00 pp | one-sided 95% +1.50 to +6.50 pp | Pass |
| Decisive-score difference | +18.81 pp | one-sided 95% +8.25 to +30.02 pp | Pass |

V6 was +0.50 pp versus both v4 and v5 in paired score, but the intervals included
zero. It produced three more avoidable repetitions than v4 and eleven more than v5.
Its decisive conditional score was essentially unchanged from v4 (-0.15 pp) and four
points below v5. The candidate therefore cannot be promoted under the frozen all-pass
rule despite being the first continuation candidate to establish positive paired
strength against off.

## Finding and next step

Expected loop value was informative: every observed short-horizon loop ended in an
available draw, so the cost of looping is exactly the advantage forgone. The remaining
problem is the binary root rule. Restoring the full original policy at 52 roots
preserved defensive options but also restored too much avoidable repetition mass.

The next target should not classify an entire root as either defense or redirect.
Instead, it should retain repeat-action mass continuously according to draw utility and
cap only the portion whose expected regret is positive. Losing/equal positions can
retain full defensive repetition mass; advantaged positions should reduce repeat mass
in proportion to `branch value - draw value`, without deleting the defensive action or
replacing the entire MCTS policy. Training length and exposure should remain frozen.

## Dashboard, artifacts, and verification

The dashboard at `http://127.0.0.1:8765/` reports corrected run `02`, step 190,
10-157-33, 151 avoidable repetitions, fixed-compute stop, strength pass, repetition
failure, and no generation.

- Corrected audit: `artifacts/audits/kasim-value-aware-risk-20260826-02/`
- Corrected candidate: `artifacts/ablations/kasim-value-aware-20260826-02/`
- Primary gate: `artifacts/evaluations/kasim-v6-exact-paired-gate.json`
- V4 comparison: `artifacts/evaluations/kasim-v6-exact-vs-v4-paired.json`
- V5 comparison: `artifacts/evaluations/kasim-v6-exact-vs-v5-paired.json`

Generated training state remains excluded from Git. The failed candidate was not
published as a champion release.

The run used the local 14-CPU-core, user-reported 32-GPU-core Apple M4 Max with 36 GiB
unified memory. MLX reported `applegpu_g16s` on macOS 26.4 arm64.

- `.venv/bin/ruff check .`: passed
- `.venv/bin/pytest -q`: 138 passed
- `npm --prefix dashboard-ui run lint`: passed
- Dashboard health and corrected snapshot assertions: passed
- Champion/new generation: unchanged/not started
