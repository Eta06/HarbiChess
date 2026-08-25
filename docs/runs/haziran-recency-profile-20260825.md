# HAZIRAN recency replay and profiler run — 2026-08-25

## Outcome

HAZIRAN re-profiled the current continuation workload, removed measured host/device
and repetition-probe overhead, separated validation from dashboard telemetry, and
trained from the unchanged champion with two recency-weighted continuation shards.
The learner pilot passed. A same-generation arena selected step 180 instead of the
minimum-validation-loss step 190. Step 180 then failed the independent 200-game
DEVIR gate, so the champion remains unchanged.

Implementation source commit: `c8dfbd1fe1396855698d2e6b69260df3dd9918e4`.

## Machine and profiler findings

Machine: Apple M4 Max 32-core GPU (`applegpu_g16s`), arm64, 36 GiB unified memory
reported by MLX, macOS 26.4.

- The actual 16-channel, 2-block network reaches about 2.4k positions/s at batch 1,
  84.9k at batch 32, 209k at batch 96, and 371k at batch 256. Raw MLX kernels were
  not the primary self-play bottleneck; Python search orchestration fragmented work.
- A 96-game, 32-simulation search benchmark reached 2,940 simulations/s with a
  37.3 average batch and 10.05 ms average queue wait. The comparable pre-change
  measurement was 2,759 simulations/s.
- 64 versus 96 full-game workers measured 59.17 versus 59.81 replay positions/s.
  Ninety-six workers is slightly faster, but the 1.1% gain and queue increase from
  15.0 to 26.2 ms show that adding workers is no longer the main lever.
- Continuation target inspection previously rebuilt/applied every visited branch.
  Reusing one isolated board and probing only the selected or policy-significant
  branches reduced its sampled time from 0.869 to 0.042 seconds per profiled game
  while preserving threefold-claim semantics and defensive repetitions.
- Host tuple to MLX conversion cost about 4.86 ms per training step. Preparing the
  replay tensors once increased the isolated learner benchmark from about 6.6k to
  21.0k positions/s and reduced a validation pass from about 407 to 6.3 ms.
- Replay sampling itself was only 0.255 seconds for 10,000 sampled batches. Replay
  encoding remains a one-time CPU cost; total non-self-play/non-training run overhead
  stayed near 23.3 seconds and is now the clearest remaining preparation bottleneck.
- In arena, candidate and champion require separate networks. The final run averaged
  batch 15.64 and 15.74, with roughly 29 ms queue wait per side. An exact 32-game
  A/B preserved identical 5-24-3 results: 0.25 ms batching took 70.08 seconds versus
  76.94 seconds at 2 ms, an 8.9% wall-time win. The split two-network inference path
  is now the dominant arena bottleneck.

## End-to-end comparison

| Metric | MAYIS | HAZIRAN | Change |
| --- | ---: | ---: | ---: |
| Self-play games | 96 | 96 | same |
| Fresh positions | 15,040 | 14,793 | workload-dependent |
| Self-play duration | 405.79 s | 316.69 s | 22.0% faster |
| Fresh positions/s | 37.06 | 46.71 | 26.0% higher |
| Training attempted steps | 234 | 310 | 32.5% more |
| Training duration | 33.93 s | 2.42 s | 14.0x faster |
| End-to-end train positions/s | 441 | 8,215 | 18.6x higher |
| Total pilot duration | 463.32 s | 342.37 s | 26.1% faster |

The self-play inference queue processed 486,683 leaf positions in 16,703 batches,
with average batch 29.14, largest batch 83, and 100.42 seconds inside the backend.

## Replay and learner

The learner combined 11,638 fresh train records with 255 versioned continuation
records. The older 125-record shard received weight 0.60 and the newer 130-record
shard weight 1.00; continuation exposure remained capped at 25% per batch. There
were no duplicate position roots between the shards. Validation remained isolated
by whole game and contained 3,155 fresh records.

Self-play produced 68 checkmates, one insufficient-material draw, and 27 max-ply
draws; no game terminated by threefold. All 96 opening prefixes were unique at ply
4, unique-game ratio was 100%, unique-position ratio 98.60%, action-space coverage
35.77%, and five policy targets were redirected away from avoidable repetition.

Train loss improved from 9.5490 to 5.4101 and validation loss from 9.5476 to 7.2913.
Training attempted 310 of 500 configured steps. It stopped because validation had
not improved for 12 evaluations / 120 steps, then atomically restored step 190.
This is not a telemetry-driven or arbitrary duration limit: validation runs every
10 steps independently of the 2-step dashboard update cadence. The dashboard shows
the exact stop reason, last improvement, patience state, restored checkpoint, and
separate arena checkpoint selection.

## Same-generation checkpoint screen

All checkpoints used the same 32 color-balanced games and opening seed `2026082532`.

| Checkpoint | Validation loss | W-D-L | Score | Elo | Continuation roots |
| --- | ---: | ---: | ---: | ---: | ---: |
| step 130 | 7.5825 | 5-20-7 | 46.88% | -21.74 | 18 |
| step 140 | 7.4108 | 3-23-6 | 45.31% | -32.67 | 20 |
| step 180 | 7.3627 | 5-24-3 | **53.13%** | **+21.74** | 21 |
| step 190 | **7.2913** | 4-22-6 | 46.88% | -21.74 | 17 |

Gameplay selected step 180. Minimum validation loss again did not identify the
best same-generation checkpoint.

## Independent DEVIR gate

Step 180 used a separate opening seed (`2026082542`) for 200 games.

- W-D-L: 13-153-34
- Score: 44.75%
- Elo estimate: -36.62
- 95% confidence interval: -59.91 to -13.64 Elo
- Terminals: 47 checkmate, 148 threefold, 5 max-ply
- Avoidable-threefold roots observed: 148
- Accepted continuation replay: 123 records (27 candidate, 96 champion)
- Mean repeating policy mass: 12.35%
- Promotion: **rejected**

The confidence interval is wholly below zero, so the champion chain remains intact.
The new 123-record target-schema-3 shard is preserved for the next generation. The
next iteration should merge it as the newest recency tier while retaining the 25%
batch cap. Because repetition improved neither score nor confidence in this gate,
the next learning-signal change should be evaluated against these newly mined roots
rather than extending the rejected candidate's training.

## Verification

- `.venv/bin/ruff check .`: passed
- `.venv/bin/pytest -q`: 119 passed
- `npm --prefix dashboard-ui run lint`: passed
- `npm --prefix dashboard-ui run build`: passed
- Dashboard live at `http://127.0.0.1:8765/`

Run artifacts occupy about 62 MiB under
`artifacts/runs/haziran-candidate-20260825-01/` and remain intentionally untracked.
