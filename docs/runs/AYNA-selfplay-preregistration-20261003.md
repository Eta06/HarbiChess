# AYNA: first self-learning phase after teacher bootstrap

Before generation. Pairwise bootstrap passed held-out learning qualification and
its first independent development arena vs dense initial: 20/40/4, score0.625,
32 opening families, paired bootstrap [0.5546875,0.6875], conservative Hoeffding
[0.384919,0.865081], one capped game. This supports a bounded learning experiment,
not strong-engine promotion. Stockfish controls are reported separately even if bad.

Hypothesis: stronger 64-simulation fresh search targets and real observed game
outcomes can improve the qualified network without erasing its useful value.
Control: frozen pairwise step2200 SHA
24679cbeacc2ad0a65e6e80ada5867bc833d60618a799270228ccf5c6800ade3.
This is an explicit weights-only new-task warm start with **optimizer reset**,
followed by full optimizer/RNG/replay/cursor resume between generations.

Run existing Torch loop, CPU FP32/one Torch thread, four concurrent game actors
through shared batching, seed20261004, 3 generations x16 games, max240 plies,
Full Gumbel64/max16 candidates/gumbel1, rolling replay3, game-balanced batch64,
64 updates/generation, all inherited/new parameters trainable, AdamW LR2e-5,
weight decay1e-4/clip5. No Stockfish actor/label during this phase. Search priors
and values both come from the current network; terminal labels come from legal
outcomes. Unknown capped outcomes remain masked, never invented draws.

First process ends after generation1 (900s ceiling), then a fresh process resumes
the full checkpoint to generation3 (1800s ceiling). Existing allocation only.
Resource/integrity/nonfinite failure stops; partial data/checkpoints retained.
Generation wall-time and actual evaluations/unique positions/updates/CPU/RSS must
be reported, not just requested search count or GPU utilization.

At generation1, if observed-terminal rows are <25% or teacher development soft-WDL
CE is >0.797972 (bootstrap0.647972+0.15), stop before generation2 and report failure.
At generation3 the same value retention gate must pass; policy drift and held-out
whole-game WDL metrics are reported too. Do not select an intermediate generation
using game-strength results. Teacher development rows are diagnostic only here,
never training targets; improvements on them alone do not establish self-learning.

Fresh independent arena: 20 unused transposition components, 40 colour-paired
games, `docs/research/AYNA-selfplay-opening-splits-20261003.json` SHA
174c94e72dcfec8fc946bc0dd34d751b00f8cbce12108445153f078a2271667d,
seed20261004, max240 plies, neural inference16 simulations/Gumbel0, one thread.
Final vs frozen bootstrap, and both vs Stockfish19/one thread/Hash16MiB/nodes32.
Self-learning strength success requires final-vs-bootstrap score>0.60, paired
bootstrap lower>0.50, caps<=10%. Report conservative uncertainty and all engine
scores; no general Elo/high-strength claim. Fixed sim/node costs differ.

If retention or strength fails, preserve the bootstrap model as development
reference, keep self-play failure failed, and register a targeted next change.
No long run until failure is explained; no threshold relaxation or repeated arena.
Learning execution and strength qualification remain distinct.
