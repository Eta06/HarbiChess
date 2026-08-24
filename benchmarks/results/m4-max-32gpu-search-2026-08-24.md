# M4 Max parallel PUCT search benchmark

Date: 2026-08-24  
Source base: `a3ea9b0` plus the uncommitted `AKIN` search implementation  
Machine: MacBook Pro `Mac16,6`, Apple M4 Max, 14 CPU cores, 32 GPU cores,
36 GB unified memory  
Software: macOS 26.4, Python 3.14.2 arm64, MLX 0.32.1  
Network: 64 channels, 4 residual blocks, BF16, compiled MLX

## Workload

Each actor searched the initial chess position for 32 PUCT simulations. Actors
had independent seeds and Dirichlet noise but shared one inference worker with a
maximum batch of 128 and a 1 ms collection deadline. MLX compilation warm-up was
excluded. This measures root-search concurrency, not complete self-play games.

```text
uv run harbichess-search-benchmark --games 16,32,64,128,256 --simulations 32
```

| Concurrent games | Elapsed (s) | Simulations/s | Mean inference batch | Largest batch |
|---:|---:|---:|---:|---:|
| 16 | 0.397 | 1,291 | 9.26 | 16 |
| 32 | 0.732 | 1,400 | 16.25 | 24 |
| 64 | 1.180 | 1,735 | 28.54 | 54 |
| 128 | 2.270 | 1,805 | 52.15 | 121 |
| 256 | 4.469 | 1,833 | 80.46 | 128 |

## Findings

- Throughput rises strongly through 64 concurrent searches.
- 128 games improve throughput by about 4% over 64 and nearly fill the maximum
  inference batch during bursts.
- 256 games improve throughput by only about 1.6% over 128 while doubling actor
  count, search trees, and wall time for the wave.
- Therefore 128 is the current root-search saturation knee, not an automatic
  full-training default. The first full self-play sweep should compare 64 and
  128 active games while the learner and replay writer are also running.
- The benchmark exposed a native MLX failure when one compiled callable crossed
  threads. The backend now keeps a compiled callable per execution thread.
