# KOPRU performance and learner-transfer report

Date: 2026-08-28  
Machine: MacBook Pro `Mac16,6`, Apple M4 Max, 14 CPU cores (10 performance + 4 efficiency), 32 GPU cores, 36 GB unified memory  
Decision: learner candidate rejected; arena and the next generation remain blocked

## Qualified replay and teacher evidence

- Fresh replay run: `kopru-qualified-replay-20260828-01`
- Replay size: 96 games, 19,254 positions (15,122 train; 4,132 validation)
- Self-play wall time: 3,585.426 s
- Clean replay/teacher alignment: verified-action advantage over raw network `+0.079969`, paired 95% interval `[+0.035808, +0.133731]`
- Stored clean target matched a fresh clean rerun exactly (TV and KL both zero).
- Noisy search was not accepted as the training target: its paired verified-action interval crossed zero.

The replay passed the frozen diversity and alignment gates. The search teacher is qualified in aggregate, but this does not mean that every individual target action is better than the raw action.

## Learner-transfer result

The valid learner transfer restored and assessed multiple validation checkpoints. Step 30 improved legal teacher-policy cross-entropy from `2.771646` to `2.689852`, but teacher top-action agreement fell from `40.71%` to `27.49%`. The no-regression top-action gate therefore rejects it. No candidate entered the arena.

Additional controlled diagnostics:

- The same regression occurs on train replay: top-action agreement `40.66% -> 27.89%`, while legal policy CE improves `2.697135 -> 2.600557`. This is not validation-only overfitting.
- Policy-only and joint policy/value training produce almost identical step-30 top-action regression. Value-head gradients are not the cause.
- A preregistered hard-top auxiliary sweep at weights `0`, `0.25`, `0.5`, and `1.0` produced no passing candidate. Higher weights also reduced tactical search solve-rate, so weight tuning stopped.
- A fixed 256-position overfit diagnostic proves that the loss and network can fit a small target set: top-action agreement rose from `40.63%` to `58.20%` at 100 steps and `80.86%` at 300 steps.
- A fixed full-replay policy-only duration diagnostic disproves simple under-training as the solution. Validation top-action agreement was `30.57%`, `30.15%`, and `29.50%` after 1, 2, and 4 epochs. Legal CE improved but the teacher argmax did not transfer.

Current diagnosis: the 16-channel, 2-block network can memorize a small subset, but the broad replay's heterogeneous, close soft visit targets cause shared-representation interference. More blind exposure or another auxiliary-loss weight sweep is not justified. The next experiment should isolate capacity/representation from target consistency using a preregistered small architecture/data matrix, while keeping this replay and arena closed to promotion.

## Live utilization profile

Measurements were taken during the 96-game KOPRU workload, not from a synthetic GPU-only loop.

- Main Python process: roughly `0.9` CPU core.
- Eight tactical-oracle worker processes: roughly `6.9-7.7` CPU cores combined.
- Effective CPU use: approximately 8 of 14 cores. Most self-play threads waited on oracle futures or the shared inference batcher.
- Apple GPU device utilization sampled from AGX counters: approximately `28-57%`; renderer `15-50%`; tiler `15-46%`. Utilization was bursty rather than saturated.
- Main-process RSS: about `1.2 GiB`; oracle workers: about `7.2 GiB` combined.
- AGX in-use memory: about `1.4-1.9 GB`.
- macOS memory pressure reported about `85%` available, with no throttled pages. Unified memory is not the bottleneck.
- `powermetrics` energy counters require elevated privileges and were unavailable; no unsupported power estimate is reported.

## Search and inference profile

Production replay generation evaluated 2,484,919 positions in 702,307 backend batches:

- Mean batch size: `3.538`
- Largest observed batch: `22`
- Backend time: `2,295.907 s`
- Aggregate queue wait: `5,813.376 s`
- Mean queue wait: `2.339 ms`

An isolated MLX masked-inference benchmark reached about `835 positions/s` at batch 1, `5,356` at batch 4, `11,454` at batch 24, and `13,398` at batch 64. Production does not reach these larger batches because the synchronous board/oracle chain feeds the GPU in small bursts.

The profiled 64-simulation search spent substantial CPU time in rule application (`1.077 s` cumulative), outcome checks (`0.721 s`), board encoding (`0.453 s`), and action conversion (`0.273 s`). MCTS child selection (`0.034 s`) and expansion (`0.051 s`) were small. Tree traversal itself is not the primary bottleneck.

Oracle worker allocation was benchmarked with the same model, 24 positions, 64 simulations, and fixed actor/batching settings:

| Workers | Simulations/s | Mean MLX batch |
|---:|---:|---:|
| 6 | 850.35 | 3.99 |
| 8 | **862.56** | **4.39** |
| 10 | 837.90 | 4.55 |
| 12 | 833.95 | 4.25 |
| 14 | 826.20 | 4.25 |

Eight workers remain the measured optimum. Adding workers raises contention and reduces wall-clock throughput; it is not a useful way to inflate CPU/GPU utilization.

## Replay loading and training profile

Before optimization, the 15,122-position train shard loaded at about `3,161 positions/s` and batch reconstruction/encoding ran at about `1,222 positions/s` (`12.378 s`). The profile showed rule validation and repeated board reconstruction dominating replay preparation.

The safe optimization shares one reconstructed board across record validation, history-aware encoding, and legal-mask generation. It does not skip checksum, schema, legality, side-to-move, policy, or continuation validation. After the change:

- Shard read: `4.726 s`, `3,200 positions/s`
- Batch preparation: `10.096 s`, `1,498 positions/s`
- Batch-preparation improvement: approximately **18.4%**

MLX learner compute is already fast once data is resident: batch-64 optimizer steps average about `3.013 ms` (`~21,242 training positions/s`). A prepared 4,132-position validation pass takes about `10.531 ms` of device compute. Reusing prepared validation encodings reduced an eight-snapshot model-quality stage estimate by about `54.8%`, and reduced the observed learner diagnostic from `62.379 s` to `51.188 s` (`17.9%`).

Therefore training GPU compute is not the wall-clock bottleneck. Replay JSON/gzip parsing, board reconstruction/encoding, oracle evaluation, and insufficiently aggregated inference requests dominate. Larger synthetic batches demonstrate headroom, but forcing dummy work or delaying search solely to raise GPU percentage would hurt the actual objective.

## Next controlled step

Do not promote, run arena, or start a new generation. Preregister a compact transfer matrix that varies only representation capacity and target consistency, records train and validation CE/top-action agreement plus tactical solve-rate, and uses the existing qualified replay. A larger architecture should proceed only if it improves fresh validation teacher imitation without regressing the frozen tactical gates. Continuous learning remains disabled until this transfer gate passes.
