# NISAN policy-iteration run — 2026-08-25

## Outcome

NISAN generated a higher-quality replay set, retained four exact validation
checkpoints, selected the strongest checkpoint by a common arena screen, and ran
an independent 200-game DEVIR gate. The candidate was not promoted. The active
champion remains unchanged.

Source commit: `16bfb3453f659a0440713a4bb34b531aa910609c`

## Learning-signal changes

- A selected move that immediately claims threefold repetition is retained when
  all visited non-repeating continuations are materially worse.
- When MCTS has a visited non-repeating continuation within `0.05` root value of
  the draw, self-play redirects both selection and the normalized replay policy
  to that continuation.
- Replay and target schemas were advanced to version 2 and record whether a
  repetition continuation was redirected.
- Training retains up to four spaced validation improvements as exact
  model/optimizer/sampler-RNG checkpoints.
- Arena can evaluate any retained checkpoint explicitly, allowing gameplay to
  choose the candidate instead of minimum validation loss alone.

## Machine and workload

- Apple M4 Max, 32-core GPU (`applegpu_g16s`), arm64
- 36 GiB unified memory reported by MLX; 28.1 GiB recommended working set
- macOS 26.4
- 96 self-play games, 64 workers, 32 MCTS simulations, 256 max plies
- 400 maximum learner steps, batch size 64, patience 20

## Replay and learner

| Metric | Result |
| --- | ---: |
| Replay positions | 14,618 |
| Self-play duration | 328.56 s |
| Self-play throughput | 44.68 positions/s |
| Training duration | 44.15 s |
| Total run duration | 394.06 s |
| Unique games | 100% |
| Unique positions | 98.66% |
| Action-space coverage | 35.87% |
| Mean policy entropy | 2.9286 |
| Repetition target redirects | 1 / 14,618 |
| Terminals | 65 checkmate, 2 insufficient material, 29 max-ply |
| Threefold terminals | 0 |

Train loss improved from 9.5490 to 4.6725. Validation loss improved from
9.5486 to 6.6688. Training attempted 258 steps, early-stopped, and exactly
restored step 218. Losses and gradients stayed finite.

## Same-generation checkpoint screen

All checkpoints used the same 32 color-balanced games, opening seed
`2026082511`, 6 opening plies, and 32 simulations.

| Checkpoint | Validation loss | W-D-L | Score | Elo | Threefold |
| --- | ---: | ---: | ---: | ---: | ---: |
| step 152 | 6.7476 | 0-27-5 | 42.19% | -54.74 | 27 |
| step 166 | 6.7436 | 2-23-7 | 42.19% | -54.74 | 23 |
| step 184 | 6.6931 | 3-23-6 | **45.31%** | **-32.67** | 21 |
| step 218 | **6.6688** | 1-26-5 | 43.75% | -43.66 | 26 |

Step 184 was selected by arena score. This directly demonstrates that minimum
validation loss was not the strongest gameplay checkpoint in this generation.

## Independent DEVIR gate

The selected step-184 checkpoint used a different opening seed (`2026082521`)
for the final 200-game gate.

- W-D-L: 13-163-24
- Score: 47.25%
- Elo estimate: -19.13
- 95% confidence interval: -39.88 to +1.48 Elo
- Terminals: 37 checkmate, 162 threefold, 1 max-ply
- All 162 threefolds had an immediately available non-repeating legal move
- Promotion: **rejected**

## Finding and next iteration

The new replay generation no longer collapsed into repetition, but deterministic
arena search still accumulated most probability mass on repeat lines. Only one
training position activated the current redirect, so the learner received too
little explicit continuation supervision. The next iteration should transform
the replay target whenever a comparable visited repeating branch holds policy
mass, even if stochastic self-play happened to select a different move. It should
also mine the arena's avoidable-threefold roots into a versioned, game-isolated
continuation replay set. Repetition must remain available when every searched
continuation is materially worse, preserving drawing defence.

## Verification

- `uv run ruff check src tests`: passed
- `uv run pytest -q`: 112 passed
- `npm run lint`: passed in `dashboard-ui`
- `npm run build`: passed; production dashboard bundle generated

The run and arena artifacts are under
`artifacts/runs/nisan-candidate-20260825-01/` and remain intentionally untracked.
