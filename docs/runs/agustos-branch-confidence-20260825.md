# AGUSTOS branch-confidence ablation — 2026-08-25

## Decision

The branch-confidence treatment is the strongest directional result so far, but it
did not clear the paired evidence threshold. Its 95% paired score-difference interval
against continuation-off still includes zero. No new generation was started and the
champion remains unchanged.

Implementation source commit: `399487e99d842700623a97e710c8dd5a18c54815`.

## Target schema v4

Target schema v4 stores the evidence used to create every confidence-gated policy:

- repeat action indices and exact repeat value;
- evaluated non-repeat action and UCI move;
- independent search sample count and simulations per sample;
- branch mean value, standard error, and lower/upper confidence bounds;
- family-wise confidence level and minimum practical advantage;
- qualified action set and exact champion model SHA-256.

Readers remain backward compatible with target-schema-3 shards. New writes use v4.
Fresh replay can have empty continuation evidence, while a confidence-gated replay
record must expose exactly the qualified actions in its policy. Defensive or uncertain
repetitions are never rewritten.

## Branch evaluation

All 378 continuation roots were reconstructed with complete repetition history. The
unchanged champion first searched each root for 128 noise-free simulations. At most
three highest-visit non-repeat branches, including the stored selected action when
needed, were then evaluated independently with 8 searches of 64 simulations each.
Each branch search used deterministic seeded root noise to measure search uncertainty.

Repeat was an immediately claimable draw, so its value was exactly 0.0 with zero
uncertainty. Non-repeat confidence bounds used a Bonferroni correction for the number
of branches compared at that root. A branch qualified only when its corrected lower
95% confidence bound exceeded `draw + 0.01`. The 0.01 practical margin prevents tiny
values such as +0.00002 from being misrepresented as meaningful improvement.

| Metric | Result |
| --- | ---: |
| Roots audited | 378 |
| Non-repeat branches evaluated | 1,134 |
| Accepted roots | 55 |
| Rejected/unchanged roots | 323 |
| Qualified branches | 62 |
| Accepted roots from NİSAN / MAYIS / HAZIRAN | 15 / 19 / 21 |
| Median qualified lower bound | +0.0151 |
| Mean qualified lower bound | +0.0631 |
| Evidence duration | 394.18 s |
| Neural leaf positions | 609,729 |
| Mean MLX batch | 44.28 |

The resulting checksummed generation-1 shard contains 55 target-schema-4 records.
Policy mass is restricted to qualified branches and weighted by confidence surplus.
The WDL target remains the observed draw; no synthetic win label is invented from a
model estimate.

## Fixed-compute training

The confidence-gated arm reused the exact TEMMUZ control conditions: unchanged
champion baseline, identical HAZIRAN fresh train/validation replay, seed `2026082504`,
200 attempted steps, batch size 64, learning rate, and best-validation restoration.
Continuation exposure stayed at the existing fixed 25% cap; no sampling-weight search
was performed.

| Treatment | Continuation target | Restored step | Validation loss |
| --- | --- | ---: | ---: |
| Off | none | 130 | 7.3258 |
| Current | 378 schema-v3 roots | 200 | **6.8526** |
| Confidence-gated | 55 schema-v4 roots | 180 | 7.2465 |

The gated run completed in 1.81 seconds. Its maximum reported pre-clip gradient norm
was 28.92; gradients were finite and clipped to 5.0 before optimizer updates.

## Matched arena

The gated candidate played the same 32-game screening seed (`2026082552`) and 64-game
confirmation seed (`2026082562`) previously used by the off/current controls. All arms
therefore share 96 identical color-balanced opening assignments and 32-simulation
search compute.

| Treatment | W-D-L | Score | Elo | 95% Elo interval |
| --- | ---: | ---: | ---: | ---: |
| Off | 1-76-19 | 40.63% | -65.92 | -96.64 to -36.20 |
| Current | 3-77-16 | 43.23% | -47.34 | -77.94 to -17.46 |
| Confidence-gated | 2-83-11 | **45.31%** | **-32.67** | -57.97 to -7.71 |

Matched paired differences:

| Comparison | Mean score difference | Better / same / worse | 95% paired interval |
| --- | ---: | ---: | ---: |
| Gated − off | +4.69 pp | 17 / 71 / 8 | -0.52 to +9.90 pp |
| Gated − current | +2.08 pp | 15 / 69 / 12 | -3.65 to +7.81 pp |

The gated arm reduced losses from 19 to 11 relative to off, but the interval narrowly
crosses zero. It also produced 80 threefold terminals versus 69 off and 76 current.
The score gain therefore came primarily from stabilizing more draws, not demonstrated
learning of non-repeating winning continuations. This is useful directional evidence
but not enough to claim the intended behavior or advance a generation.

## Next evidence requirement

Keep schema v4 and the 0.01 confidence gate. Before another generation, run a larger
paired confirmation or improve the branch evaluator with rollout outcomes that can
distinguish genuine winning conversion from model-value calibration. A future gate
should require both a positive paired strength interval and no regression in avoidable
threefold rate. Merely increasing training length or continuation exposure is not
justified by this result.

## Verification and artifacts

- `.venv/bin/ruff check .`: passed
- `.venv/bin/pytest -q`: 128 passed
- `npm --prefix dashboard-ui run lint`: passed
- Branch evidence: `artifacts/audits/agustos-branch-evidence-20260825-02/branch-evidence.json`
- Schema-v4 shard: `artifacts/audits/agustos-branch-evidence-20260825-02/continuation-confidence-gated.jsonl.gz`
- Gated ablation: `artifacts/ablations/agustos-gated-20260825-01/`
- Dashboard remained live at `http://127.0.0.1:8765/`
