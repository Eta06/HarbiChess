# MART profiler optimization, candidate, and arena — 2026-08-25

## Scope and environment

- Apple M4 Max (`applegpu_g16s`), 32-core GPU variant
- 36 GiB unified memory; MLX 0.32.1 on native arm64 Python 3.14
- optimized candidate source: `c0baecda697df3530655d44df407c34092088eba`
- final dashboard/report source includes later telemetry corrections through MART

The work used cProfile call-time data, fixed network/adapter microbenchmarks,
matched end-to-end self-play, and a worker-count sweep. MLX compilation and
evaluation choices follow the official MLX guidance: compile reusable graphs,
avoid shape churn, and remember that `.item()`/Python conversion synchronizes
lazy GPU work.

## Baseline profiler findings

The original 8-game, 64-ply, 12-simulation profile recorded about 145 million
Python calls in 33.75 seconds.

- `PythonChessRules.board`: 17.05 cumulative seconds
- repeated `python-chess` move pushes: 1.46 million calls
- legal membership checks: about 1.24 million calls
- encoding: 10.37 cumulative seconds
- MLX backend active self-time: only about 1.76 seconds

The small 16-channel/2-block network itself reached about 156,461 positions/s
at batch 128 and 220,472 positions/s at batch 256. The GPU was therefore being
starved by state reconstruction, encoding, legal masking, Python traversal, and
small/serialized inference batches rather than limited by M4 Max arithmetic.

## Implemented optimizations

1. Bounded thread-local incremental `python-chess.Board` cache. Every immutable
   `ChessState` retains full legal/repetition history; public boards remain
   mutation-isolated.
2. Legal masking and encoding share the same read-only board. Encoder history
   copies retain only the seven prior boards needed by the eight-step feature
   stack, while repetition is computed on the full-history board.
3. Bounded immutable encoded-state cache reuses a selected child when it becomes
   the next search root.
4. Encoder plane offsets use python-chess's already canonical square index
   directly instead of recomputing rank/file for every write.
5. MLX gathers only requested legal policy logits before GPU-to-Python transfer.
   The previous path transferred all 4,672 logits per position and discarded
   almost all of them.
6. Prepared immutable replay batches are reused. Validation replay is no longer
   reconstructed every telemetry interval, and sampled batches select already
   validated rows.
7. Gradient finiteness is reduced as one MLX expression instead of synchronizing
   once per parameter tensor. The optimizer still cannot update on a non-finite
   gradient.
8. Exact best-validation snapshots include model, optimizer, learner step, and
   sampler RNG. Early stopping restores all four before the atomic candidate
   checkpoint is written.
9. Self-play repetition collapse has its own guardrail. Arena telemetry reports
   whether a threefold choice had a non-repeating legal alternative without
   changing chess rules or biasing the match.

## Matched before/after result

The SUBAT workload and `mart-matched-benchmark-01` used the same seed, initial
model, 32 games, 32 workers, 12 simulations, and 256-ply ceiling. They produced
the exact same 6,114 positions, W/D/L, terminal distribution, and diversity
metrics.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Self-play time | 289.33 s | 49.69 s | 5.82x faster |
| Self-play throughput | 21.13 pos/s | 123.04 pos/s | 5.82x |
| Batch-128 backend adapter | 5,814 pos/s | 17,238 pos/s | 2.97x |
| Fixed batch-32 train step | 4,037 pos/s | 5,575 pos/s | 1.38x |

The matched terminal identity is important: speed did not come from truncating
games, reducing simulations, changing random streams, omitting repetition
history, or weakening legal validation.

## Worker sweep

The fixed 64-game, 64-ply, 12-simulation sweep measured:

| Workers | Positions/s | Mean inference batch | Games/hour |
|---:|---:|---:|---:|
| 8 | 96.0 | 5.91 | 5,563 |
| 16 | 117.5 | 9.80 | 6,807 |
| 24 | 123.1 | 12.45 | 7,134 |
| 32 | 131.1 | 17.38 | 7,598 |
| 48 | 131.3 | 17.89 | 7,610 |
| 64 | 138.3 | 31.31 | 8,016 |

After the final encoder cache, the 64-worker workload improved again to 147.94
positions/s and 8,573 games/hour. Sixty-four is the measured optimum among the
tested configurations for this compact network/workload. It is not a universal
constant: a larger network, longer search, memory pressure, or fewer available
games can move the optimum. The 48-to-64 gain was modest, so blindly adding
threads beyond the available game wave would add memory and scheduling cost
without creating a larger inference batch.

## MART candidate

Run `mart-candidate-20260824-01` used 64 games, 64 workers, 24 simulations per
move, a 256-ply ceiling, batch 64, and a 320-step maximum learner budget.

- 10,260 replay positions; 7,572 train and 2,688 validation
- self-play: 184.11 seconds
- training: 8.62 seconds
- 41 checkmates, 21 max-ply draws, 2 threefold repetitions
- 64/64 unique games; 98.66% unique positions
- 32.47% action-space coverage; 14.77 effective policy branches
- initial validation loss: 9.5845
- best validation loss: 7.8021 at step 28
- training attempted through step 60, then early stopping restored step 28
- verified candidate SHA-256:
  `fbcf98f090d65e760c9c5b669b8c98516a385ac92c20d93601ec16597d864ced`

The run performed roughly four times the SUBAT self-play search work (twice the
games and twice the simulations) yet completed self-play in 184 seconds versus
289 seconds.

## DEVIR screen and decision

Arena `mart-devir-screen-20260824-01` used 32 color-balanced opening pairs, 24
simulations, 64 workers, and the exact stored baseline.

- candidate: 4 wins / 53 draws / 7 losses
- score: 47.656%
- Elo estimate: -16.30
- 95% interval: -52.01 to +19.07 Elo
- terminations: 11 checkmates, 2 max-ply draws, 51 threefold repetitions
- all 51 threefold repetitions had a non-repeating legal alternative
- promotion: rejected; champion remains unchanged

The screen has no positive promotion signal, so expanding it to the full
200-game gate would waste compute. The candidate remains an auditable failed
challenger. Best validation loss is now selected correctly, but this result also
shows that supervised replay validation improvement is not a substitute for
arena playing strength.

## Remaining bottlenecks and next direction

The final profile is dominated by feature-plane construction, python-chess board
copy/push, legal generation, repetition-claim analysis, and MCTS traversal. Raw
MLX inference capacity remains much higher than end-to-end throughput. Further
large gains require a structural actor redesign: persistent per-game boards and
trees, a compact native feature representation, and possibly a vectorized/native
legal-move path. Those changes are higher risk and must retain the current rules,
perft, repetition, perspective, deterministic RNG, and matched-game tests.

For learning quality, the next candidate should not simply train longer on this
replay. It should improve policy iteration signal—more useful search targets,
repetition-aware training examples, and evaluation of multiple saved validation
checkpoints—while the baseline/champion chain remains unchanged.
