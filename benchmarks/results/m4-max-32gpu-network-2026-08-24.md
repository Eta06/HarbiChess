# HarbiChess MLX Network Benchmark — M4 Max 32-core GPU

## Reproducibility

| Field | Value |
|---|---|
| Date | 2026-08-24 |
| Source commit | `e14d4b02c4055f67e94541972b4f465a62a26a81` |
| Computer | MacBook Pro `Mac16,6` |
| SoC | Apple M4 Max |
| CPU | 14 cores: 10 performance + 4 efficiency |
| GPU | 32 cores, Metal 4 |
| Unified memory | 36 GB |
| MLX recommended working set | 30,150,672,384 bytes |
| macOS | 26.4 (`25E241`) |
| Python | 3.14.2, native arm64 |
| MLX | 0.32.1 |
| Dtype | bfloat16 |
| Network | 104 input planes, 64 trunk channels, 4 residual blocks |
| Policy | 4,672 fixed actions (`8×8×73`) |
| Parameters | 2,769,551 |
| Compilation | `mx.compile` enabled |
| Warm-up | 3 iterations per batch |
| Measured iterations | 50 per batch |

Command:

```bash
uv run harbichess-network-benchmark \
  --batches 1,8,16,32,64,128,256,512 \
  --iterations 50
```

## Results

| Batch | Mean latency | Positions/second |
|---:|---:|---:|
| 1 | 0.753 ms | 1,328 |
| 8 | 0.667 ms | 11,985 |
| 16 | 0.711 ms | 22,518 |
| 32 | 0.709 ms | 45,107 |
| 64 | 0.814 ms | 78,617 |
| 128 | 1.216 ms | 105,271 |
| 256 | 1.991 ms | 128,549 |
| 512 | 3.562 ms | 143,756 |

## Findings

1. Batch 1 is dominated by dispatch and fixed inference overhead.
2. Batch 64–128 provides a strong latency/throughput balance for the first
   shared self-play evaluator.
3. Batch 512 maximizes isolated network throughput, but it is not automatically
   the best self-play batch. Compared with batch 128 it gains about 36.6% more
   positions/second while mean inference latency grows about 2.9×.
4. An MCTS game normally cannot advance a dependent search path until its leaf
   evaluation returns. With 64 active games, filling a batch of 512 requires an
   average of eight in-flight leaves per game or additional waiting. That can
   reduce completed moves and games per hour even while raw GPU throughput rises.
5. Large batches also reserve more unified memory and create longer tail latency
   when training kernels share the same GPU. Desktop responsiveness is not the
   limiting argument; search dependency latency and queue fill time are.

## Current concurrency conclusion

Among the requested logical concurrency candidates, the network-only evaluator
has enough capacity for 128 active game slots. This is not yet an end-to-end
self-play optimum. The final answer must be selected by a 16/32/64/128-game MCTS
benchmark using games/hour, nodes/second, batch fill ratio, and p95 leaf latency.
The initial shared evaluator policy will target batches of 64–128 with a short
latency deadline instead of waiting indefinitely for the maximum batch.

