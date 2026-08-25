# ARALIK continuous value-regret ablation — 2026-08-26

## Decision

The v7 candidate passed the fresh avoidable-threefold, win-rate, and decisive-score
guardrails, but did not establish positive paired strength with 95% confidence. It was
not promoted and no new generation was started. Training length, continuation
exposure, and target temperature were not changed after observing results.

## Why the arena data changed

The prior 200 opening/color assignments had been reused across several target
iterations and had therefore become development feedback. ARALIK retained all frozen
gate thresholds and compute settings but pre-registered a fresh paired arena before
playing games:

| Block | Games | Seed | Workers |
| --- | ---: | ---: | ---: |
| Screen | 32 | `2026082582` | 32 |
| Confirm | 64 | `2026082592` | 64 |
| Evidence | 104 | `2026082602` | 96 |

Both continuation-off and v7 played the same fresh assignments. Every game used 12
opening plies, 32 MCTS simulations, a 256-ply limit, and a 0.25 ms inference wait.
This required 400 new arena games but prevented further optimization against the old
test set.

## Continuous policy target

Target schema v7 preserves readers for target schemas 3 through 6 and adds a versioned
policy-regret record containing root value, draw/repeat value, best risk-adjusted
non-repeat value, regret, temperature, blend fraction, action sets, and champion hash.

The source continuation replay had direct immediate-repeat policy mass at only one of
55 roots. Scaling only those actions would therefore have been ineffective. Instead,
v7 continuously blends the broad original continuation policy with the safe v6
redirect policy:

`regret = max(0, min(root value, best non-repeat value) - draw value)`

`redirect fraction = 1 - exp(-regret / 0.02)`

`target = (1 - fraction) × original policy + fraction × safe redirect policy`

This conservative minimum requires both the root and a non-repeat branch to show an
advantage. Losing/equal roots have zero regret and retain their original policy.
Positive regret moves mass gradually rather than switching an entire root.

## Replay audit

| Metric | Result |
| --- | ---: |
| Replay records | 55 |
| Zero-regret defensive roots | 13 |
| Continuously adjusted roots | 42 |
| Mean redirect fraction | 33.45% |
| Median redirect fraction | 42.18% |
| Maximum redirect fraction | 92.79% |
| Direct repeat mass before / after | 3.23% / 1.29% |

The direct repeat-mass total is small because earlier continuation generation already
removed most immediate repeat actions; the larger effect is reducing mass on broad
policies that can return to a loop later.

## Fixed-compute training

The candidate used the unchanged champion, HAZIRAN train/validation replay, seed
`2026082504`, 200 attempted steps, batch size 64, 55 continuation records, and 25%
continuation exposure. It reached the fixed maximum-step limit and restored the best
validation checkpoint at step 110.

| Metric | Result |
| --- | ---: |
| Training time | 1.80 s |
| Initial / final train loss | 9.54885 / 6.40528 |
| Initial / best validation loss | 9.54764 / 7.62549 |
| Maximum gradient norm | 20.33199 |

## Fresh paired arena

| Arm | W-D-L | Score | Avoidable threefold | Decisive score |
| --- | ---: | ---: | ---: | ---: |
| Fresh continuation-off | 5-166-29 | 44.00% | 162 / 200 | 14.71% |
| V7 continuous regret | 11-160-29 | **45.50%** | **157 / 200** | **27.50%** |

V7 did not merely turn losses into draws: losses were unchanged at 29 while wins rose
from five to eleven. It also reduced avoidable repetitions by five games.

| Guardrail | Estimate | Confidence interval | Result |
| --- | ---: | ---: | --- |
| Paired score difference | +1.50 pp | two-sided 95% -2.75 to +5.75 pp | **Fail** |
| Avoidable-threefold difference | -2.50 pp | one-sided 95% -8.50 to +3.50 pp | Pass |
| Win-rate difference | +3.00 pp | one-sided 95% 0.00 to +6.50 pp | Pass |
| Decisive-score difference | +12.79 pp | one-sided 95% -3.49 to +28.83 pp | Pass |

The frozen decisive rule checks non-regression in the point estimate, so it passes;
its interval is still uncertain. The all-pass promotion rule fails solely because the
paired strength lower bound remains below zero.

## Finding and next step

Continuous value regret corrected the KASIM repetition regression without erasing the
directional strength signal. However, KASIM's old-set strength pass did not replicate
with confidence on fresh assignments. Temperature tuning against this new result would
immediately turn the fresh set into another development set and is not justified.

The clean next experiment is to keep this exact v7 checkpoint and target frozen and
collect additional pre-registered fresh paired arena evidence. That tests whether the
+1.5-point strength estimate is real without increasing training time, continuation
exposure, or changing the model. Promotion remains blocked until strength confidence
and all behavioral guardrails pass together.

## Dashboard and artifacts

The dashboard at `http://127.0.0.1:8765/` reports run
`aralik-value-regret-20260826-01`, restored step 110, 11-160-29, 157 avoidable
repetitions, repetition pass, strength uncertainty, and no generation.

- Replay audit: `artifacts/audits/aralik-value-regret-20260826-01/`
- Candidate: `artifacts/ablations/aralik-value-regret-20260826-01/`
- Fresh control arenas: `artifacts/ablations/temmuz-off-20260825-01/arena/aralik-off-*`
- Gate: `artifacts/evaluations/aralik-v7-fresh-paired-gate.json`

Generated training state remains excluded from Git. The unpromoted candidate was not
published as a champion release.

The run used the local 14-CPU-core, user-reported 32-GPU-core Apple M4 Max with 36 GiB
unified memory. MLX reported `applegpu_g16s` on macOS 26.4 arm64.

- `.venv/bin/ruff check .`: passed
- `.venv/bin/pytest -q`: 141 passed
- `npm --prefix dashboard-ui run lint`: passed
- Dashboard health and final snapshot assertions: passed
- Champion/new generation: unchanged/not started
