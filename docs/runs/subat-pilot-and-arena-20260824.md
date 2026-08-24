# SUBAT second pilot and DEVIR arena — 2026-08-24

## Environment

- Apple M4 Max (`applegpu_g16s`), 32-core GPU variant
- 36 GiB unified memory; MLX recommended working set about 28.1 GiB
- macOS 26.4 arm64
- source commit: `cc0e36985fc79f294fc312777b0ca8d0aa7e8cf3`

## OCAK pilot

Run `subat-pilot-20260824-04` started from the exact persisted random baseline
and used 32 parallel games, 12 MCTS simulations per move, a 256-ply ceiling,
and 160 learner steps with batches of 32.

- result: passed every sanity guardrail
- self-play: 32 games and 6,114 positions in 289.33 seconds
- learner: 160 steps in 217.00 seconds; total run 531.65 seconds
- replay split by whole game: 26 train games / 6 validation games
- replay samples: 4,901 train / 1,213 validation
- train loss: 9.5917 -> 6.3723 (33.6% reduction)
- validation loss: 9.6253 -> 8.2415 (14.4% reduction)
- best observed validation loss: 7.6414 at step 106
- maximum gradient norm: 15.187; no NaN or non-finite failure
- verified checkpoint: `candidate-step-000160`

The final validation loss remained better than baseline, but rose after step
106 while train loss continued to improve. This is early overfitting evidence;
blindly extending this replay's training budget is not the next experiment.

## Diversity and terminal behavior

- 32/32 unique games; 98.95% unique positions
- all 32 opening prefixes were unique at plies 4, 8, and 12
- 1,333 selected action indices; 28.53% global action-space coverage
- mean policy entropy 2.0466; 7.74 effective policy branches
- 16 checkmates (10 white wins, 6 black wins)
- 16 max-ply draws
- zero self-play threefold repetitions

The larger search/replay budget fixed the first pilot's low decisiveness: half
of the games ended in checkmate. The remaining 50% max-ply rate is still high
enough to monitor, but is below the configured 90% rejection threshold.

## Exact model lineage

- baseline SHA-256: `90d6b15730790223b0c6469ca5ee9b1f4dc8729898131d24d34268cc7c3beef1`
- candidate SHA-256: `f8d63949c758dbe0c12808a25764acfbc0daf094d05a5690931f369fed888353`
- candidate checkpoint integrity: verified
- complete run artifact footprint: 19 MiB

The stored baseline, candidate model, optimizer, sampler RNG, replay manifests,
and schema hashes make the run resumable and auditable without reconstructing
the baseline from a seed.

## DEVIR arena

Arena `subat-devir-20260824-01` used 16 color-balanced opening pairs (32
games), four randomized opening plies, 12 simulations per move, and 32 workers.
It compared the candidate with the exact persisted baseline.

- candidate: 1 win / 30 draws / 1 loss
- score rate: 50.0%
- estimated Elo delta: 0.0
- 95% interval: -30.66 to +30.66 Elo
- terminations: 2 checkmates and 30 threefold repetitions
- promotion: rejected; champion remains unchanged
- elapsed: 54.93 seconds

The arena's 93.75% repetition rate contrasts with zero repetition in self-play.
This points to deterministic evaluation collapse between two nearly equal weak
policies, not replay/opening collapse: exploratory self-play remained diverse,
while arena action selection repeatedly chose the same cycles. Running the full
200-game promotion gate would spend compute without a positive signal and would
not narrow an exactly neutral result into evidence of improvement.

## Decision and next step

Keep the existing champion/baseline and retain this candidate only as an
auditable failed challenger. Before a third larger training run:

1. select the checkpoint at the best validation step (or add early stopping),
2. add repetition-aware search behavior so known repetition cycles do not absorb
   deterministic arena play when non-drawing alternatives exist,
3. optimize self-play's Python board/tree hot path and inference batching, then
4. generate a fresh, larger replay generation and train a new candidate from the
   unchanged champion.

This preserves the champion chain, addresses the arena failure mode directly,
and avoids rewarding lower training loss that did not produce playing strength.
