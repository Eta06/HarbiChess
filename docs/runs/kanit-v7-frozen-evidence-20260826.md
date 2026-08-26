# KANIT frozen v7 evidence — 2026-08-26

## Decision

Frozen v7 did not pass the pre-registered combined 600-game gate. Repetition,
win-rate, and decisive-performance guardrails passed, but the paired strength interval
still included zero. V7 was not promoted, no new generation was started, and the
champion chain remains unchanged.

## Pre-registration integrity

The complete design was committed and pushed as `c8e267e` before any additional game
was played. It fixed the following without interim adaptation:

- exactly 400 additional games per arm;
- 200 color-balanced pairs at seed `2026082612`;
- combination with all existing 200 ARALIK games for exactly 600 games per arm;
- 12 opening plies, 32 simulations, 256 maximum plies, 96 workers, and 0.25 ms wait;
- 50,000 paired bootstrap samples with seed `2026082603`;
- the existing strength, avoidable-threefold, win-rate, and decisive guardrails.

The V7 and continuation-off model files were SHA-256 verified before and after the
pre-registration. The 400 new control/candidate games had identical pair indices,
candidate colors, and opening moves. No training, exposure, replay, temperature,
policy, checkpoint, threshold, or sample-size change occurred.

## Additional 400-game block

| Arm | W-D-L | Score | Arena time |
| --- | ---: | ---: | ---: |
| Continuation-off | 6-325-69 | 42.13% | 884.22 s |
| Frozen V7 | 14-318-68 | 43.25% | 904.50 s |

The additional block was not evaluated in isolation for promotion. It was combined
with the complete pre-registered ARALIK block.

## Combined 600-game evidence

| Arm | W-D-L | Score | Avoidable threefold | Decisive score |
| --- | ---: | ---: | ---: | ---: |
| Continuation-off | 11-491-98 | 42.75% | 468 / 600 | 10.09% |
| Frozen V7 | 25-478-97 | **44.00%** | **467 / 600** | **20.49%** |

V7 gained fourteen additional wins while losses decreased by one. This is not a
loss-to-draw-only effect, and its decisive conditional score improved by 10.40 points.
Avoidable repetition improved by one game.

| Guardrail | Estimate | Confidence interval | Result |
| --- | ---: | ---: | --- |
| Paired score difference | +1.25 pp | two-sided 95% -1.08 to +3.58 pp | **Fail** |
| Avoidable-threefold difference | -0.17 pp | one-sided 95% -3.83 to +3.50 pp | Pass |
| Win-rate difference | +2.33 pp | one-sided 95% +0.67 to +4.00 pp | Pass |
| Decisive-score difference | +10.40 pp | one-sided 95% +2.53 to +18.15 pp | Pass |

The decisive improvement is now supported by a positive one-sided interval, not just
a non-regressing point estimate. Nevertheless, the all-pass rule fails solely on
paired strength confidence.

## Interpretation

The larger independent sample refined the V7 effect from +1.50 to +1.25 percentage
points and narrowed uncertainty, but not enough to establish positive paired strength.
The observed effect is below the approximately +3.4-point practical effect used to
size this 600-game experiment. Extending the sample after observing the result would
violate the pre-registration, so no more games were added.

V7 remains an informative unpromoted checkpoint: continuous value regret corrected
the KASIM repetition regression, increased wins, and improved decisive performance.
It does not yet meet the required evidence standard for champion promotion. A future
iteration should seek a larger strength effect through a new pre-registered target or
search change, then use another independent arena set; it should not tune against these
600 assignments.

## Dashboard and artifacts

The dashboard at `http://127.0.0.1:8765/` reports 600 combined games, 25-478-97,
467 avoidable repetitions, repetition/decisive pass, strength failure, and no
promotion or generation.

- Pre-registration: `docs/runs/kanit-v7-arena-preregistration-20260826.md`
- Additional off arena: `artifacts/ablations/temmuz-off-20260825-01/arena/kanit-off-additional/`
- Additional V7 arena: `artifacts/ablations/aralik-value-regret-20260826-01/arena/kanit-v7-additional/`
- Combined gate: `artifacts/evaluations/kanit-v7-combined-600-paired-gate.json`

Generated arena state remains excluded from Git. No champion release was published.

## System

The evidence run used the local 14-CPU-core, user-reported 32-GPU-core Apple M4 Max
with 36 GiB unified memory. MLX reported `applegpu_g16s` on macOS 26.4 arm64.

## Verification

- Frozen model SHA-256 checks: passed.
- 400 fresh paired opening/color assignments: passed.
- `.venv/bin/ruff check .`: passed.
- `.venv/bin/pytest -q`: 141 passed.
- `npm --prefix dashboard-ui run lint`: passed.
- Dashboard health and combined snapshot assertions: passed.
- Champion/new generation: unchanged/not started.
