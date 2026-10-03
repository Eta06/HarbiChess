# AYNA legal-only pairwise inference optimization

Before implementation/benchmark. Qualified pairwise72,497 parameters is smaller
than dense1,256,355, but the current masked method computes all4672 policy scores
before selecting legal moves. Measured candidate-vs-SF32 arena throughput is
459.4 network evaluations/wall-second; smaller is not automatically faster.

Hypothesis: score only requested origin/destination/plane tuples in masked
inference, retaining the exact full learner forward/parameters and root-STM WDL.
This is an execution optimization, not a new learned model. Same weights SHA
24679cbeacc2ad0a65e6e80ada5867bc833d60618a799270228ccf5c6800ade3.
Full and masked outputs must agree within atol2e-5/rtol2e-5 on both real MLX CPU
and PyTorch, including legal special moves; old architecture tests stay passing.
No gate changes to primary arenas, which use the already-loaded previous source.

Benchmark original source48a1f0f in isolated worktree and optimized source, same
real initial-position encoding/legal actions, warmup5/repeats30, batch1/8/32,
threads1/2/4 through existing Torch benchmark. Timed regions include backend
tensor construction, masked forward and host outputs; encoding is outside.
Run sequentially with no arena/training concurrently. Report batch latency,p95,
positions/s; primary speed gain criterion >=15% one-thread batch1 without
parity failure. Thread/batch generalization or overall self-play acceleration
requires its own wall measurement. No GPU/Apple Metal speed claim.
