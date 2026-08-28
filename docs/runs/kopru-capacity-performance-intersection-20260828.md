# KOPRU capacity and performance intersection

Date: 2026-08-28  
Quality matrix source commit: `f313004cf8f0b75287eb1f762fb7f26fc915e565`  
Decision: no learner architecture qualified; arena and new generation remain blocked

## Frozen capacity result

All variants were function-preserving expansions of the same release baseline, trained policy-only for exactly 474 steps (two replay-equivalent epochs), with batch 64 and sampler seed `2026082814`. The maximum initial output delta was `9.69e-8`, below the preregistered `1e-5` limit.

| Variant | Parameters | Validation legal CE | Teacher top agreement | Raw tactical | Search-64 tactical | Train positions/s |
|---|---:|---:|---:|---:|---:|---:|
| Release baseline, untrained | 1,229,305 | 2.771646 | **40.71%** | 1/8 | 6/8 | n/a |
| Base: 16ch, 2 blocks, policy 4 | 1,229,305 | **2.616706** | 29.72% | 0/8 | 6/8 | 6,907 |
| Deep: 16ch, 4 blocks, policy 4 | 1,238,585 | 2.615958 | 29.94% | 0/8 | 6/8 | 6,499 |
| Head: 16ch, 2 blocks, policy 8 | 2,425,405 | 2.621369 | 29.43% | 0/8 | 6/8 | 6,800 |
| Deep-head: 16ch, 4 blocks, policy 8 | 2,434,685 | 2.619486 | 29.48% | 0/8 | 6/8 | 6,333 |

No variant passed. Additional depth and a doubled policy head did not restore teacher argmax transfer, and every trained variant lost the release baseline's one raw tactical solve. Capacity alone is therefore not the primary blocker. The existing release architecture remains the deployment choice because the expanded models provide no learning-quality gain to offset their cost.

The result strengthens the target-consistency diagnosis: search is stronger in aggregate, but the broad replay's per-state soft targets do not form a stable, transferable improvement signal for the learner. The next quality experiment must measure clean-teacher consensus across budgets and verified per-state improvement/confidence before changing architecture, exposure, or loss weights again.

## Frozen search performance result

The accepted code optimization reuses an already validated cached child transition when MCTS traverses the same state and move again. It removes repeated `python-chess` board copy, UCI parse, legality check, and push work while preserving the immutable `ChessState` and cached board isolation.

Same checkpoint, 24 positions, 64 simulations, four repeats, eight oracle workers, 24 actors, batch cap 48, and 0.25 ms batching wait:

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Mean wall time | 1.7702 s | 1.6410 s | **-7.30%** |
| Search throughput | 868.7 sim/s | 937.2 sim/s | **+7.89%** |
| Mean MLX batch | 4.41 | 4.34 | -1.7% |
| Mean backend time | 1.1907 s | 1.0928 s | -8.2% |
| Mean queue wait | 2.449 ms | 2.217 ms | -9.5% |

An eight-position real model + depth-1 oracle equivalence diagnostic compared the legacy and cached paths. Selected moves, complete visit distributions, priors, Q values, and root values were bitwise identical; maximum root-value delta was zero.

## Scheduling and batching

The preregistered batch-wait sweep measured:

| Wait | Throughput | Mean batch |
|---:|---:|---:|
| 0 ms | 572.4 sim/s | 1.00 |
| 0.10 ms | 930.8 sim/s | 4.12 |
| **0.25 ms** | **940.9 sim/s** | 4.26 |
| 0.50 ms | 927.8 sim/s | 4.46 |
| 1.00 ms | 892.8 sim/s | 5.25 |

The current 0.25 ms wait remains optimal. Waiting longer raises GPU batch size but reduces end-to-end throughput.

A 48-root microbenchmark favored 16 actors, but the production-shaped dual-search mini self-play contradicted the microbenchmark. On the same 24 game seeds, 16 plies, 64 simulations, clean/noisy dual search, and depth-1 oracle:

| Workers | Wall time | Games/hour | Positions/s | Mean MLX batch |
|---:|---:|---:|---:|---:|
| 16 | 32.064 s | 2,694.6 | 11.98 | 5.07 |
| **24** | **28.347 s** | **3,047.9** | **13.55** | **6.99** |

All 24 game trajectories were identical between configurations. The production setting therefore remains 24 game workers and eight oracle workers. This is a concrete example of why utilization or a root-only microbenchmark cannot override end-to-end games/hour.

## Utilization after the accepted optimization

During an eight-repeat, 48-root workload:

- Search throughput: 1,047.8 simulations/s; mean MLX batch 5.09.
- AGX device utilization: 60-63%; renderer 54-58%; tiler 50-55%.
- Main Python process: approximately 0.7-1.1 CPU core.
- Eight oracle processes were active, typically using roughly 0.35-0.82 core each in the sampled intervals.
- AGX in-use memory stayed around 1.72 GB; memory capacity was not a bottleneck.

GPU utilization is higher than the earlier 28-57% range, but the objective remains wall-clock throughput. Increasing wait to create batch 5.25 reduced throughput, and increasing worker count beyond the production optimum increased contention.

## Rejected optimization

A sparse legal-policy head computed only requested legal logits instead of all 4,672 outputs. It matched selected full-policy logits within `1e-5` in float32 and reduced backend time by about 9.4%, but total search changed from 868.7 to 866.6 simulations/s. The CPU/oracle/batching pipeline absorbed the device-side saving, so the production backend was not changed. The isolated network primitive remains available for future larger networks, where the full policy head may become material.

## Intersection decision

The performance winner is the cached-transition code with the existing `24 game workers / 8 oracle workers / 0.25 ms wait` configuration. There is no quality winner: all four capacity variants fail the frozen transfer gate. Consequently, combining a new architecture with the performance configuration and running final qualification would be post-hoc and invalid. The release baseline plus accepted performance optimization is retained, while learner, arena, promotion, and generation stay disabled pending a separately preregistered teacher-target consistency experiment.
