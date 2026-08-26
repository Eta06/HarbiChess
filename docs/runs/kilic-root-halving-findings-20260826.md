# KILIC root sequential-halving findings — 2026-08-26

## Decision

KILIC achieved its mechanical objective—rare, high-margin root changes at the same
neural-evaluation budget—but failed the expected-score, win-rate, and decisive-game
promotion guardrails. Avoidable repetition improved. The candidate remains weaker
than the unchanged champion, so no promotion or new generation was started.

Implementation and generation source commit:
`85dbf64564aa642513d1f5d136c68def7c08a8f0`.

## Fixed-compute verification

The control used one root expansion plus 64 ordinary MCTS simulations. KILIC used one
root expansion plus 32 broad simulations, four forced top-action searches of three
simulations, and two finalist searches of seven simulations. A real MLX smoke run
measured exactly 520 evaluations for eight roots (`8 × 65`), matching the control's
65-evaluation root cost.

No ATAK value-temperature target, continuation replay, or prior replay was used. Both
arms started from champion SHA-256
`5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`,
used fresh seed `2026082617`, and shared the pre-registered game-level split namespace.
Replay target schema 9 preserved the root adjustment flag and both confidence margins.

## Fresh replay and learner

Machine: Apple M4 Max 32-core GPU (`applegpu_g16s`), arm64, 36 GiB unified memory
reported by MLX, macOS 26.4.

| Metric | 64-sim control | KILIC root halving |
| --- | ---: | ---: |
| Games / positions | 96 / 14,420 | 96 / 15,870 |
| Decisive games | 64 | 59 |
| Max-ply draws | 30 | 35 |
| Repetition redirects | 5 | 3 |
| Neural evaluations | 933,461 | 1,016,890 |
| Average inference batch | 28.25 | 30.85 |
| Self-play seconds | 702.69 | 661.72 |
| Training seconds | 2.53 | 3.01 |
| Attempted steps | 320 | 400 |
| Restored best-validation step | 200 | 280 |

The neural-evaluation totals differ only because KILIC trajectories contained 1,450
more replay positions. Per ordinary root the budget remained matched. Both learners
stopped after 12 validation evaluations / 120 steps without improvement and restored
their best-validation checkpoints.

KILIC evaluated 15,486 eligible roots and adjusted only 59 (`0.381%`). Mean final
margin on adjusted roots was 0.958. This is the intended sparse/high-confidence shape,
not ATAK's small widespread reweighting.

## Why the high-confidence signal did not generalize

The 59 adjusted records came from 56 games. Their median ply was 103 (range 7–234),
only three occurred by ply 30, and 13 occurred at ply 150 or later. From each recorded
side-to-move perspective, outcomes were 56 wins, two draws, and one loss.

The confidence gate therefore selected mostly already-winning, often late tactical or
terminal-resolution branches. Large margins were real but not broadly useful policy
improvements. Sharpening those easy decisions reduced neither strategic uncertainty
nor the candidate's generalization error; fresh self-play also became less decisive
and produced five more max-ply draws.

## Same-generation checkpoint screen

All meaningful validation checkpoints played 32 games on seed `2026082620`. These
games were excluded from final evidence.

| Arm | Step | W-D-L | Score |
| --- | ---: | ---: | ---: |
| Control | 140 | 2-25-5 | **45.31%** |
| Control | 160 | 1-26-5 | 43.75% |
| Control | 180 | 1-22-9 | 37.50% |
| Control | 200 | 1-25-6 | 42.19% |
| KILIC | 220 | 2-26-4 | **46.88%** |
| KILIC | 230 | 1-25-6 | 42.19% |
| KILIC | 250 | 1-23-8 | 39.06% |
| KILIC | 280 | 2-25-5 | 45.31% |

Gameplay selected control step 140 and KILIC step 220 instead of either arm's
minimum-validation-loss checkpoint. Their model SHA-256 values are respectively
`70d8da6d709a780420fc7a46db5952d523ae6aa50c787973546d2a2102e9c993` and
`8107560191452b1eb4a6595bac3a953e1f0c3edd79f3566a84ea8c8e0092b1bc`.

## Fresh 200-game paired arena

Both selected models played the champion on 100 new color-balanced opening pairs
from seed `2026082618`. The paired gate used 50,000 bootstrap samples and seed
`2026082619`.

| Metric | Control | KILIC | KILIC minus control |
| --- | ---: | ---: | ---: |
| W-D-L | 8-163-29 | 9-154-37 | — |
| Expected score | 44.75% | 43.00% | -1.75 pp |
| Elo estimate | -36.62 | -48.96 | — |
| Elo 95% interval | [-57.12, -16.36] | [-71.79, -26.55] | — |
| Threefold / avoidable | 157 / 157 | 149 / 149 | -4.00 pp avoidable |
| Checkmate / max-ply | 37 / 6 | 46 / 5 | — |

Frozen gate results:

- Strength: -1.75 pp, two-sided 95% interval `[-5.75, +2.25]` — **failed**.
- Minimum +4.00 pp effect and absolute champion superiority — **failed**.
- Avoidable-threefold: -4.00 pp, one-sided interval `[-11.00, +3.00]` — passed.
- Win rate: +0.50 pp, one-sided interval `[-2.50, +3.50]` — **failed**.
- Conditional decisive score: 21.62% to 19.57%, -2.06 pp — **failed**.

The repetition improvement is genuine but cannot compensate for worse expected score,
more losses, and regressed conditional decisive performance. The champion chain and
generation remain unchanged.

## Next technical direction

Do not tune KILIC's margin, transfer fraction, exposure, or temperature on this arena.
A future search-target iteration should distinguish *decision-changing regret* from
easy terminal confidence: qualify only roots where deeper search overturns or
materially improves upon the original visit leader/selected action, explicitly
exclude trivial resolved wins from policy transfer, and measure value gain against
the original action rather than winner-versus-runner separation alone. That requires
a new keyword, pre-registration, fresh replay, and fresh arena.

## Artifacts and verification

- Pre-registration: `docs/runs/kilic-root-halving-preregistration-20260826.md`
- Control: `artifacts/runs/kilic-control-20260826-01/`
- KILIC: `artifacts/runs/kilic-root-halving-20260826-01/`
- Paired gate: `artifacts/evaluations/kilic-root-halving-fresh-200-paired-gate.json`
- Generated run footprint: approximately 126 MiB and intentionally excluded from Git.
- Dashboard: final KILIC arena, selected step 220, early-stopping reason, W-D-L,
  repetition counts, and unchanged champion are live at `http://127.0.0.1:8765/`.

Verification after the experiment:

- `.venv/bin/python -m ruff check .`: passed.
- `.venv/bin/python -m pytest -q`: 151 passed.
- `npm --prefix dashboard-ui run lint`: passed.
- `npm --prefix dashboard-ui run build`: passed.
