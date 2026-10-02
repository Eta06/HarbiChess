# PORT: Linux execution and fresh policy iteration preflight — 2026-10-02

Authorization: user's 2 October request permits Linux implementation, existing-budget training,
commits/push and Space updates. No paid resources or release/champion promotion.
Starting source: f4bdc85638b18cb313278567ceabf99fb21798a0; work branch, clean checkout.
Stage prefix PORT means portable execution and measured policy iteration, not success.

Hypothesis H1: a strict FP32 PyTorch implementation can preserve the shared NHWC encoder,
4672-action mapping, side-to-move WDL order (win/draw/loss), masked CE and unknown-outcome
mask while using the existing rules, batched search and replay loader on Linux CPU.
Control: independent NumPy NHWC forward with MLX-shaped weights; all base/residual branches,
including MIHVER/DENGE, must match within atol=2e-5/rtol=2e-5. Reject unknown keys/shapes.
Roundtrip model weights losslessly; conversion is weights-only warm start, never optimizer resume.
Real MLX/Metal and CUDA parity are untested without those devices; do not add skips to hide failures.

H2: optimizer, RNG, immutable replay checksums and completed-generation cursor support an
actual separate-process resume with identical subsequent sampled indices, loss and weights on
this CPU in deterministic FP32. Corrupt artifacts/config/replay must fail before training.
Checkpoint at generation boundaries; an interrupted uncommitted generation restarts from the
last complete boundary. Keep all earlier artifacts; never overwrite a published checkpoint.

H3 (smoke, not strength qualification): fresh search targets/outcomes can feed repeated network
updates, retaining rolling replay and optimizer. Compare fixed replay loss before/after 32 steps;
finite parameter changes and loss decrease establish execution, not generalization or Elo.
Run at most 3 generations × 8 games × 192 plies × 16 simulations, 32 steps/generation, batch 32;
seed 20261002; game-balanced sampling, replay window 3 generations. Held-out whole-game
metrics and terminal-observed fraction reported. Unknown max-ply targets stay masked.
Use fresh policy targets from current weights; no oracle training labels or retrospective thresholds.
Stop on nonfinite values, integrity failure, 30 minutes total wall time or resource limit.
No long training until this works; 4 CPU quota, 16 GiB RAM, no detected GPU; allocation only.

Measure warmed inference positions/s and latency at batch 1/8/32, actual search/game wall time,
sampled training rows/s separately from unique/generated positions, peak RSS and CPU time.
Benchmark CPU threads 1/2/4 on identical weights/positions; choose fastest observed setup for
this allocation, then freeze it. Do not extrapolate GPU utilization or Apple benchmark speed.

Strength diagnostic: initial versus latest, uniform legal random, Stockfish single-thread nodes=32
(Hash=16 MiB; no pondering; record engine version). Eight predefined opening families, both
colors per family, max 192 plies, deterministic Full Gumbel 16 for neural engine; seed fixed.
Record capped games separately from true draws, W/D/L, opening-pair scores and uncertainty.
Small 16-game comparisons cannot establish general Elo; no promotion on this evidence.
Tactical frozen checks: mate in one and terminal sign, black perspective, castling, en passant,
promotion, history/repetition through existing rule/action tests and real backend search.

Research: re-read primary AlphaZero, Mctx Gumbel, Lc0 and KataGo mechanisms; map evidence
to CPU budget decisions. Preserve DENGE failure. Later-only backlog: external Laya (link requested,
identity unknown), search/decision assistance cost, high-quality dataset, stronger student via new
search/RL/outcome evidence, LLM legal play/strength/explanation measured independently.
