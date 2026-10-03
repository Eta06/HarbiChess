# AYNA pairwise bootstrap and first independent games

## Decision and measured strength

The compact origin/destination policy passed its registered learning and first
development game-strength gates. It is a **development reference**, not a strong
engine/champion. It still has no wins against even32-node Stockfish19.

| Candidate/opponent | Wins / true draws / losses / unknown caps | Diagnostic score | Paired bootstrap95% | Wall seconds |
|---|---|---:|---|---:|
| Pairwise / original dense initial | 20 /39 /4 /1 | 0.625000 | [0.554688,0.687500] |220.3350|
| Pairwise / Stockfish32 |0 /20 /44 /0|0.156250|[0.109375,0.210938]|68.7895|
| Original dense / Stockfish32 |0 /11 /53 /0|0.085938|[0.046875,0.132813]|45.1511|

One capped game has an unknown result; score0.5 is diagnostic only. Even scoring
that game as a loss gives0.617188, above the0.60 point gate. First row conservative
Hoeffding95% [0.384919,0.865081] is wide and includes0.5. Fixed-suite bootstrap
supports further controlled work, not universal Elo or a guaranteed effect.
Against the same engine, paired score delta0.0703125, bootstrap
[0.0078125,0.1328125]. Still 44/64 engine losses; the main strong-engine goal is
unmet. Increasing compute or doing more identical updates is not automatically
supported by this result.

All three arms used32 previously unseen transposition-separated opening families,
both colours, seed20261003, max240 plies; network16 FullGumbel simulations,
Gumbel0, one Torch CPU thread. Stockfish19 one thread/Hash16MiB/requested32nodes.
These are unequal-cost sim/node budgets, not equal-wall comparisons. Actual engine
nodes were not captured by this arena interface. Network evaluations114325/31602/
23313, rates518.9/459.4/516.3 per arena-wall-second. Different game lengths affect
cost; no speed comparison inferred from whole-arena duration alone.

Independent verifier replayed all192 games/14881 plies, checked legality,
opening histories, candidate-colour scores, observed endings and cap identity.
No game/seed rerun or checkpoint selection from arena. Original first-two source
48a1f0f298e2ed517ca9fcc0e2f5502b43bad1f9; control source is captured in its raw
report (intervening commits only preregistration/opening documentation).

## Learning mechanism, controls and failure preserved

[Registration](AYNA-pairwise-preregistration-20261003.md). Conventional relational/
attention scorer; no originality claim. Remove dense policy, retain immutable
trunk/value from step200 of the failed all-parameter arm; new policy and optimizer
reset explicitly. Direct current-piece/global/coordinate features and destination
interaction avoid the historical origin-only limitation. Only16914 new policy
parameters trained; total72497 versus1256355 previously (17.33x fewer).

Training source30180e6223d353aca3ad5f41353712f7c17eb1da, seed20261003,
AdamW2e-4, batch64, eval every200, same2605train/996validation rows. Early stop
at3200, best2200 selected solely by development validation. Validation has already
guided the failed prior arm and is not independent strength evidence.

| Metric | Original dense initial | Selected pairwise2200 |
|---|---:|---:|
| Policy CE |3.272551|2.966929|
| Native soft-WDL CE |0.876855|0.647972|
| Reference Q MAE |0.575775|0.332212|
| Teacher best move in top16 |0.597390|0.799197|

Frozen inherited WDL stayed identical throughout this policy training. No claim
that this ablation isolates all strength gains to policy: value transfer and policy
both differ from the original dense control. Future mechanistic strength ablations
and multiple training seeds are needed. Failed dense1200-updates arm remains
failed, with every optimizer/model checkpoint retained.

Training wall89.5334s, selfCPU91.4028s including startup, peak569196KiB.
Initial architecture-transfer SHA beb64c90cbdfef40ab127f4ec51085a082de2e7eb7ea87c0daf96a6f5bcfd7a5;
selected model24679cbeacc2ad0a65e6e80ada5867bc833d60618a799270228ccf5c6800ade3;
oracle dataset33640698d2462894376b48d7afbed76ddd6b3a20544bd543985ed395690c3cd4;
opening split019c5b4a49ac287ca03effb65b33ed97daa2e702ddd4a6972491969e27912b24.
Full raw commands/metadata/checkpoints/replay are in local `artifacts/ayna-*`.

## Backend compatibility and measured speed

Version1 weights carry an explicit `pairwise` specification; older readers must
reject this unknown architecture. Strict tensor layout conversion and new MLX
class preserve shared104/4672/STM-WDL semantics. Old architectures/replay/weights
are unchanged. New architecture transfer is weights-only; full optimizer/RNG/
dataset/cursor checkpoints are separate. Old bootstrap CLI resumes require its
recorded source checkout because its code-hash guard rejects changed trainer code;
this refusal is intentional, not a weights-only substitute for full resume.

Real MLX CPU/PyTorch full/masked forward, soft-target loss and gradients passed,
including black perspective, castling, en-passant and underpromotion geometry.
Apple Metal/CUDA devices were unavailable; no device performance/resume claim.
MLX AdamW bias-correction differs from Torch default; optimizer updates are not
claimed equivalent. Full suite before masked optimization:482passed,0skipped.
After mask optimization:12 related tests passed; final broader result is recorded
in the completed stage ledger.

[Masked speed registration](AYNA-masked-speed-preregistration-20261003.md): same
weights in isolated old-source worktree versus optimized code, no concurrent
arena/training. One-thread batch1:910.4 ->1086.4 positions/s (1.193x), batch8:
1910.4 ->2331.0, batch32:2366.4 ->2830.8. Timed tensor construction/masked forward/
host outputs included, encoding excluded,5warmups/30repeats. Selected dot products
reassociate FP32 sums: numerical tolerance2e-5, not bitwise parity or identical
search trajectories promised. All primary arena processes loaded pre-optimization
code; subsequent self-play uses the committed optimized implementation.

## Preservation and next experiment

Raw games/speed logs/models/datasets remain local. Small results/evidence hashes
are retained in Git; complete generated binary artifacts belong in Releases.
Upload remains blocked: a genuine439-byte ASCII inventory over negotiated HTTP/2
also returned400 BadContent-Length, like priorHTTP/1.1/gh attempts. No remote
backup success claim. No new paid compute, no old data removed.

[True self-play registration](AYNA-selfplay-preregistration-20261003.md) freezes
48games/three generations, stronger search targets, lowLR, retention and fresh
20-family arena gates. Generation1 actually generated16games/2765positions,
10observed endings/1325known-outcome rows,178273 network evaluations in380.2386s.
64updates then preserved teacher WDL CE0.644260 (baseline0.647972), passing the
registered0.797972 retention ceiling; teacher policy CE worsened to3.009100.
One held-out game's observed WDL CE1.803853 ->1.710644 is descriptive, not strength.
Generation1 checkpoint exists; later learning/strength outcomes must not be
assumed from execution. Threaded CPU utilization motivates a separate spawn
throughput test, not a silent change to this registered learning run.
