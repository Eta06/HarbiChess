# AYNA: spawned CPU self-play versus shared threaded search

Before implementation/measurement. Current four-thread64-simulation self-play
uses about one CPU (process sampling), despite four allocated cores. This is
consistent with Python rule/search GIL work, but not a profiler proof of cause.

Hypothesis: independent spawned CPU actor processes with one Torch thread and
one small immutable model snapshot each can improve real game-generation wall
throughput. Preserve the shared threaded inference option for larger models/GPU.
This is CPU process parallelism, not model strength or an original RL mechanism.

Freeze pairwise bootstrap weights SHA
24679cbeacc2ad0a65e6e80ada5867bc833d60618a799270228ccf5c6800ade3,
seed20261004, eight games from standard position,64-ply cap,16-simulation
Full Gumbel/max16/gumbel1, four actors/one Torch thread, no training/oracle.
Compare threaded control then spawn implementation once each,120-second ceiling
per arm. Record actual legal games, plies, network evaluations, startup-inclusive
wall time, sum process/children CPU and RSS. Whole-game records must be restored
and legal, caps remain unknown. Hash both raw outputs. Four-thread batching can
change FP32 association; do not claim identical trajectories across modes unless
actual records prove it. Success: >=1.5x generated positions/wall-second with no
rule/target/provenance failure. Run without concurrent training/arena.

Process mode is explicit and CPU-only, recorded in full checkpoint configuration.
Default threaded run_config remains unchanged so existing thread checkpoints
still resume. Cross-mode/config resume must be refused; a changed experiment
requires a declared new run/transfer, not silent reinterpretation of old state.
Old MLX execution and old replay/checkpoints remain intact. A failed/slow spawn
arm remains failed. Further long learning decisions depend on quality gates.
