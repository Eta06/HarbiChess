# OCAK sanity runs — 2026-08-24

## Environment

- Machine: MacBook Pro `Mac16,6`
- SoC: Apple M4 Max, 14 CPU cores, 32-core GPU variant
- Unified memory: 36 GB
- MLX device: `applegpu_g16s`
- macOS: 26.4 arm64

## First run: diagnostic rejection

`ocak-sanity-20260824-04` used source commit
`b9ddd175dbec1ea5151142251086dc02af4219bc`, 12 games, 8 MCTS
simulations, a 64-ply limit, and 40 learner steps.

The numerical loss gate passed: training loss fell from 9.6532 to 5.7539 and
validation loss from 9.6499 to 6.7546. However, all 12 games reached the
64-ply cap and were labelled draws. The zero value loss therefore measured a
draw-only target collapse rather than useful WDL learning. This exposed a
missing guardrail. The run must not be used for arena or promotion decisions.

The runner was changed to require decisive terminal outcomes and to cap the
allowed max-ply draw ratio. The dashboard now reports both measurements.

## Second run: accepted sanity pilot

`ocak-sanity-20260824-08` used source commit
`e56b536c4cda43d8edfd6c90ddeda76984f0e6bf` with:

- 16 parallel games, 12 train and 4 validation
- 4 MCTS simulations per move
- 192-ply limit
- 40 MLX learner steps, batch size 16
- compact 16-channel, 2-block pilot network

The run completed in 89.81 seconds: 47.68 seconds of self-play and 32.22
seconds of training. It produced 2,828 replay positions, including 751 held-out
validation positions.

| Measurement | Result |
| --- | ---: |
| Train loss | 9.6366 → 7.5374 (-21.8%) |
| Validation loss | 9.6138 → 8.4844 (-11.7%) |
| Maximum pre-clip gradient norm | 9.3847 |
| Decisive games | 3 / 16 (18.75%) |
| Max-ply draws | 13 / 16 (81.25%) |
| Unique games | 100% |
| Unique positions | 99.08% |
| Selected action-space coverage | 21.10% |
| Effective policy branches | 2.84 |
| Unique opening prefixes at ply 4/8/12 | 16 / 16 / 16 |

The candidate checkpoint `candidate-step-000040` was written atomically,
checksummed, loaded into a fresh learner, and verified. Its model, optimizer,
RNG, replay cursor, and manifest occupy approximately 14 MB. Replay shards add
approximately 116 KB.

## Decision

OCAK passes as a pipeline sanity check: policy and non-trivial WDL targets train
without non-finite values, held-out loss improves, self-play has no observed
opening or full-game duplication, and exact resume works. This does not establish
playing strength. The correct next step is a small color-balanced DEVIR arena
against the unchanged baseline. Promotion must remain disabled unless the
predefined confidence rule passes.
