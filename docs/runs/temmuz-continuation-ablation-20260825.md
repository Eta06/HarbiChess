# TEMMUZ continuation signal audit and ablation — 2026-08-25

## Decision

No new generation was started. The existing continuation treatment was directionally
best in the matched arena, but its paired confidence interval still includes zero.
Neither the current treatment nor the filtered treatment demonstrated a reliable
strength increase over continuation-off. The champion remains unchanged.

Implementation source commit: `6dc83f3d9e7828cfdaf19f845ceeac45aa9d4297`.

## Audit method

The audit covered all 378 versioned continuation records: 125 from NİSAN mining,
130 from the MAYIS gate, and 123 from the HAZIRAN gate. Every root was reconstructed
with full repetition history and re-searched by the unchanged champion for 128
noise-free MCTS simulations, four times the 32-simulation generation budget.

For every root the audit records:

- champion root value and its change from the stored root value;
- claimable-repeat moves, repeat visit mass, and repeat MCTS value;
- stored non-repeat target visit mass and visit-weighted MCTS value;
- whether the champion's most-visited move repeats;
- whether the stored target contains the champion's selected non-repeat move;
- source outcome from the side-to-move perspective.

A target is search-supported only when the champion selects a non-repeat move that
the stored target contains, the target has at least four visits and 10% visit mass,
and its visit-weighted value is within 0.02 of the guaranteed repetition draw. A
target is harmful when it retains repeat overlap, is more than 0.05 below the repeat
value, or points away from a different non-repeat move selected by the champion.
Champion-selected repetition without contrary evidence is quarantined as uncertain.

## Audit findings

| Source | Supported | Uncertain | Harmful | Total |
| --- | ---: | ---: | ---: | ---: |
| NİSAN roots | 4 | 109 | 12 | 125 |
| MAYIS roots | 6 | 88 | 36 | 130 |
| HAZIRAN roots | 5 | 104 | 14 | 123 |
| **Total** | **15** | **301** | **62** | **378** |

All source outcomes were draws, so outcome alone could not justify redirection.
The champion still selected repetition at 300 of the 301 uncertain roots. Across
the 15 supported roots, stored targets held 67.86% mean visit mass and the champion
selected a target non-repeat move at every root. Their mean target value was -0.0086,
which is search-aligned but not evidence that the continuation is stronger than the
guaranteed draw. The filtered shard therefore represents the least ambiguous subset,
not proven winning positions.

The audit processed 45,791 neural leaf positions in 35.57 seconds, with average MLX
batch 42.4. It emitted a checksummed generation-1 shard containing only the 15
supported records. Harmful and uncertain records remain in the audit JSON but are
excluded from that shard.

## Fixed-compute training ablation

All three arms used the same unchanged champion baseline, the exact same HAZIRAN
fresh train/validation split, seed `2026082504`, learning rate, network, 200 attempted
steps, batch size 64, and best-validation checkpoint restoration.

- **Off:** no continuation examples.
- **Current:** all 378 records, generation/recency weights, 25% batch exposure.
- **Filtered:** only 15 supported records. Exposure was 0.992%, calculated as
  `25% × 15 / 378`, so each retained record receives the same expected exposure as
  a record in the current arm instead of being oversampled hundreds of times.

| Treatment | Continuation | Restored step | Train loss | Validation loss | Training time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 0 / 0% | 130 | 5.6869 | 7.3258 | 2.09 s |
| Current | 378 / 25% | 200 | 5.3873 | **6.8526** | 2.00 s |
| Filtered | 15 / 0.992% | 190 | **5.1359** | 7.1004 | 1.99 s |

Loss ranking was deliberately not used as the strength decision.

## Matched arena ablation

Each arm played the same 96 color-balanced games against the unchanged champion:
32 screening games with seed `2026082552` and 64 confirmation games with independent
seed `2026082562`. Search used 32 simulations and identical opening/max-ply settings.

| Treatment | W-D-L | Score | Elo | 95% Elo interval |
| --- | ---: | ---: | ---: | ---: |
| Off | 1-76-19 | 40.63% | -65.92 | -96.64 to -36.20 |
| Current | 3-77-16 | **43.23%** | **-47.34** | -77.94 to -17.46 |
| Filtered | 4-69-23 | 40.10% | -69.68 | -106.34 to -34.50 |

Matched per-game bootstrap comparisons:

| Comparison | Mean score difference | Better / same / worse | 95% paired interval |
| --- | ---: | ---: | ---: |
| Current − off | +2.60 pp | 17 / 66 / 13 | -3.13 to +8.33 pp |
| Filtered − off | -0.52 pp | 16 / 63 / 17 | -6.25 to +5.21 pp |
| Current − filtered | +3.13 pp | 21 / 60 / 15 | -3.13 to +9.38 pp |

The current method ranks first directionally, but none of the paired intervals
excludes zero. All three candidates are also confidently weaker than the champion.
Consequently, selecting current as a proven winner or advancing a new generation
would violate the requested strength guardrail.

## Recommended next experiment

Do not add more repetitions of the same continuation target. The audit shows the
main issue is evidence quality: 79.6% of roots still lead the champion to choose the
repeat draw, while only 4.0% are aligned with a supported non-repeat target. The next
small experiment should regenerate these roots with deeper search or a short rollout
after each repeat/non-repeat branch and store branch-level value uncertainty in a
new target schema. Only roots whose non-repeat lower confidence bound exceeds the
repeat draw should enter training. That tests a stronger signal instead of merely
changing sampling weights.

## Verification and artifacts

- `.venv/bin/ruff check .`: passed
- `.venv/bin/pytest -q`: 124 passed
- `npm --prefix dashboard-ui run lint`: passed
- Audit artifact: `artifacts/audits/temmuz-continuation-20260825-02/audit.json`
- Filtered shard: `artifacts/audits/temmuz-continuation-20260825-02/continuation-reliable.jsonl.gz`
- Ablation artifacts: about 14 MiB per arm, intentionally untracked
- Dashboard remained live at `http://127.0.0.1:8765/`
