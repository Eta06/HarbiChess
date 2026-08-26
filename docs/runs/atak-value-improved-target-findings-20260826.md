# ATAK value-improved target findings — 2026-08-26

## Decision

ATAK improved the matched control's expected score and preserved all repetition,
win-rate, and decisive guardrails, but the strength gain was too small and uncertain
for promotion. The candidate also remained significantly weaker than the unchanged
champion. The champion chain and generation therefore remain unchanged, and no new
generation was started.

Implementation and experiment source commit: `ff5247798e1461336e924a2281c4dad6f79d7ec0`.

## Treatment and replay audit

The target reweighted MCTS visit counts with a bounded, eight-visit-shrunk
root-relative action-value advantage. It ran after the existing repetition-safe
transformation and after move selection, so it could not change legal behavior or
self-play trajectories. Replay target schema 8 distinguishes the new target.

Both arms started from champion SHA-256
`5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`,
used seed `2026082613`, and independently generated 96 games at 64 simulations. No
old replay or continuation shard was used. The resulting 12,796 positions had
identical states, selected actions, and outcomes. Two final-ply root values differed
only by `0.6e-6` and `2.5e-6`, consistent with concurrent batched floating-point
ordering.

The treatment changed policy probabilities in 12,759 of 12,796 records. Mean policy
total-variation distance was only 0.00975 (0.98 percentage points), although the
maximum was 0.4210. This explains the central result: the transformation was safe and
occasionally substantial, but shallow root action values were usually too similar to
provide the intended large strength signal.

## Self-play and learner

Machine: Apple M4 Max 32-core GPU (`applegpu_g16s`), arm64, 36 GiB unified memory
reported by MLX, macOS 26.4.

| Metric | Visit control | Value-improved target |
| --- | ---: | ---: |
| Games / positions | 96 / 12,796 | 96 / 12,796 |
| Decisive / max-ply | 74 / 20 | 74 / 20 |
| Repetition redirects | 16 | 16 |
| Neural leaf evaluations | 827,556 | 827,556 |
| Average inference batch | 25.35 | 25.33 |
| Self-play seconds | 586.33 | 560.74 |
| Training seconds | 2.24 | 2.17 |
| Attempted training steps | 390 | 370 |
| Best-validation step | 270 | 250 |
| Early-stop reason | no improvement for 120 steps | no improvement for 120 steps |

The two run IDs caused deterministic train/validation membership to differ, so their
raw validation losses are not compared across arms. Checkpoint selection used the
pre-registered same-generation gameplay screen rather than validation loss alone.

## Same-generation checkpoint screen

Every meaningful validation checkpoint played 32 games on screen seed `2026082616`.
These games were excluded from the final evidence.

| Arm | Step | W-D-L | Score |
| --- | ---: | ---: | ---: |
| Control | 140 | 2-28-2 | **50.00%** |
| Control | 160 | 2-24-6 | 43.75% |
| Control | 210 | 0-26-6 | 40.63% |
| Control | 270 | 1-28-3 | 46.88% |
| Treatment | 80 | 2-20-10 | **37.50%** |
| Treatment | 100 | 1-21-10 | 35.94% |
| Treatment | 170 | 0-23-9 | 35.94% |
| Treatment | 250 | 1-19-12 | 32.81% |

Control step 140 and treatment step 80 were frozen for the independent final arena.
Their model SHA-256 values are respectively
`8f5fdd703f643d7f267cc604c10284e99c4d8ab7698921aa25e563d9753d6fd6` and
`4aa25ce1ca7d5ae6e80879fee07885558d6cd252d593f87a7f9dcebf6e2ae607`.

## Fresh 200-game paired arena

Arena seed `2026082614` supplied 100 new color-balanced opening pairs. Each model
played the unchanged champion at 32 simulations; bootstrap seed `2026082615` used
50,000 paired resamples.

| Metric | Control | Treatment | Treatment minus control |
| --- | ---: | ---: | ---: |
| W-D-L | 2-170-28 | 8-169-23 | — |
| Expected score | 43.50% | 46.25% | +2.75 pp |
| Elo estimate | -45.42 | -26.11 | — |
| Elo 95% interval | [-63.47, -27.61] | [-44.96, -7.40] | — |
| Threefold / avoidable | 170 / 170 | 161 / 161 | -4.50 pp avoidable |
| Checkmate / max-ply | 30 / 0 | 31 / 8 | — |

Paired gate results:

- Strength: +2.75 pp, two-sided 95% interval `[-1.00, +6.50]` — **failed**.
- Pre-registered minimum effect: +4.00 pp — **failed**.
- Absolute score above champion with lower bound above 50% — **failed**.
- Avoidable-threefold: -4.50 pp, one-sided interval `[-10.50, +1.50]` — passed.
- Win rate: +3.00 pp, one-sided interval `[+0.50, +5.50]` — passed.
- Conditional decisive score: 6.67% to 25.81%, +19.14 pp — passed.

The behavior gain was not merely losses converted to draws: wins increased from two
to eight, losses fell from 28 to 23, and conditional decisive score improved. It is
still not promotion evidence because paired strength includes zero and the treatment
remains below the champion with a wholly negative Elo interval.

## Next technical direction

Do not increase this treatment's exposure or tune its temperature on the completed
arena. A subsequent iteration should obtain stronger action discrimination from the
search itself: reserve a fixed root budget for confidence-aware top-action
re-evaluation (for example, sequential-halving or forced top-k continuation search),
then train on those completed-search values. The current bounded target can remain a
fallback only when value confidence is adequate. Repetition handling should remain a
separate guardrail. That proposal requires a new pre-registration, fresh replay, and
fresh arena; ATAK supplies no authority to promote or start that generation.

## Artifacts and verification

- Pre-registration: `docs/runs/atak-value-improved-target-preregistration-20260826.md`
- Control: `artifacts/runs/atak-control-20260826-01/`
- Treatment: `artifacts/runs/atak-value-policy-20260826-01/`
- Paired gate: `artifacts/evaluations/atak-value-policy-fresh-200-paired-gate.json`
- Combined run footprint: approximately 127 MiB; generated state remains excluded
  from Git.
- Dashboard: final ATAK arena, selected step 80, training stop reason, W-D-L,
  threefold/avoidable counts, and unchanged champion are live at
  `http://127.0.0.1:8765/`.

Verification completed after the experiment:

- `.venv/bin/python -m ruff check .`: passed.
- `.venv/bin/python -m pytest -q`: 145 passed.
- `npm --prefix dashboard-ui run lint`: passed.
- `npm --prefix dashboard-ui run build`: passed.
