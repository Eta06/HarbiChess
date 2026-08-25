# MAYIS continuation replay run — 2026-08-25

## Outcome

MAYIS converted avoidable-threefold arena roots into versioned training replay,
broadened repetition-aware policy transformation beyond the selected move, and
trained a new candidate from the unchanged champion. Four exact validation
checkpoints were screened by gameplay. Step 150 was selected instead of the
minimum-validation-loss step 194, then failed the independent 200-game DEVIR
promotion gate. The champion remains unchanged.

Implementation commit: `b5a19795cb6aa6fe3621999c2077ed8d5d2c2fe2`

## Target and replay guardrails

- A repeat branch is considered without being selected when it owns at least
  10% of visited MCTS policy mass.
- Repeat mass is removed only if a visited non-repeat continuation is within
  0.05 root value of the best repeat branch.
- When all searched continuations are materially worse, repetition remains a
  valid drawing defence.
- Continuation replay uses replay schema 2, target schema 3, legal-action
  validation, whole-game train split, atomic gzip writing, and SHA-256 payload
  verification.
- Continuation examples are capped at 25% of each learner batch so mined roots
  cannot overwhelm fresh self-play.
- The candidate initializes from the current champion weights instead of a new
  random network.

## NİSAN root mining

The deterministic NİSAN step-184 arena was replayed with the original 200-game
configuration. It reproduced 13-163-24 and yielded:

- 162 avoidable-threefold roots
- 125 accepted continuation targets
- 37 protected repetition defences because alternatives were materially worse
- 18 candidate roots and 107 champion roots
- Mean original repeating policy mass: 7.68%
- Payload SHA-256:
  `2500beaa95221bc8853148bd38c7369ce4779bcc3a7ebca59a322fee6272a676`
- Source commit: `71df928e32dabd236985b6bfe3021c639fe7a8cb`

## MAYIS generation and learner

Machine: Apple M4 Max (`applegpu_g16s`), arm64, 36 GiB unified memory reported
by MLX, macOS 26.4.

| Metric | Result |
| --- | ---: |
| New self-play games | 96 |
| New positions | 15,040 |
| Mined continuation records | 125 |
| Total learner records | 15,165 |
| Self-play duration | 405.79 s |
| Training duration | 33.93 s |
| Total duration | 463.32 s |
| Unique games | 100% |
| Unique positions | 98.72% |
| Action-space coverage | 36.26% |
| Self-play target redirects | 2 |
| Terminals | 65 checkmate, 3 insufficient material, 28 max-ply |
| Threefold terminals | 0 |

Train loss improved from 9.5490 to 5.1680. Validation loss improved from
9.5486 to 7.0686. Training attempted 234 steps, early-stopped, and restored the
exact model/optimizer/sampler state at step 194. Losses and gradients remained
finite.

## Same-generation checkpoint screen

All checkpoints used the same 32 color-balanced games and opening seed
`2026082531`.

| Checkpoint | Validation loss | W-D-L | Score | Elo | Continuation roots |
| --- | ---: | ---: | ---: | ---: | ---: |
| step 124 | 7.2532 | 3-23-6 | 45.31% | -32.67 | 17 |
| step 136 | 7.1926 | 0-28-4 | 43.75% | -43.66 | 21 |
| step 150 | 7.1309 | 1-30-1 | **50.00%** | **0.00** | 24 |
| step 194 | **7.0686** | 2-27-3 | 48.44% | -10.86 | 25 |

Gameplay selected step 150, demonstrating again that minimum validation loss is
not a sufficient checkpoint selector.

## Independent DEVIR gate

Step 150 used a separate opening seed (`2026082541`) for 200 games.

- W-D-L: 17-159-24
- Score: 48.25%
- Elo estimate: -12.17
- 95% confidence interval: -34.06 to +9.63 Elo
- Terminals: 41 checkmate, 157 threefold, 2 max-ply
- New continuation replay: 130 records
- Root sources: 43 candidate, 87 champion
- Mean repeating policy mass: 15.07%
- Promotion: **rejected**

Compared directionally with NİSAN's separate-seed gate, score improved from
47.25% to 48.25%, Elo from -19.13 to -12.17, wins from 13 to 17, and threefolds
from 162 to 157. Because the opening seeds differ, this is an encouraging trend,
not proof of strength improvement.

The failed candidate did not alter the champion chain. Its 130-record,
target-schema-3 continuation shard is preserved for the next generation. The
next iteration should combine the prior and new shards with generation-aware
recency weighting, while keeping total continuation exposure bounded. A larger
paired screen is warranted before changing target tolerance or claiming that
repetition has materially improved.

## Verification

- `uv run ruff check src tests`: passed
- `uv run pytest -q`: 115 passed
- `npm run lint`: passed in `dashboard-ui`
- `npm run build`: passed
- Dashboard remained live at `http://127.0.0.1:8765/`

Run artifacts are under `artifacts/runs/mayis-candidate-20260825-01/` and remain
intentionally untracked.
