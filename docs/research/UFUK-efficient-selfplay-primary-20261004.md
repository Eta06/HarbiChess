# UFUK: recent self-play efficiency work and concrete decisions

4 October2026 primary-source research during the unchanged broad-v4collection.
Provider HuggingFace paper_search returned UNAVAILABLE twice. Direct official
arXiv API and HTML search returned200; three full HTML papers were fetched,
hash-recorded and converted to text for focused methods/results/limitations
reading. The paired source evidence preserves exact retrieved snapshots, not
an independent reproduction of their claims. No weights/code were imported,
no paid resource obtained and no active experiment rule changed.

## Efficient self-play system, September2026

Bertil Braun, [Engineering Efficient Self-Play Chess: Search, Replay, and
Throughput Under Limited Compute](https://arxiv.org/html/2609.37447v1),
arXiv2609.37447v1/27September2026. Exact HTMLSHA
824f15abee5b6756c84e585eba4fc55f926b221c5ed98c99aa70efea952666a7.

Author-reported system:6.32Mparameter14block/160channel CNN,52planes,
1880from-to policy actions, WDL, next-policy and remaining-length auxiliaries.
Training2.5days on8RTX4070SUPER GPUs and80logicalCPUs,3.25Mcompletedgames,
209Mmaterializedpositions/836.608Mtrainingpresentations. Estimated100Bsimulations
uses assumed500freshsimulations/position, explicitly not independently measured.
Reported3251benchmarkElo[3206,3297] uses100000searches/move and a historical
fixed-node Stockfish13 ladder; it is not a universal Elo or our SF19comparison.
The paper counts300plyevaluation caps as draws; our declared cap bounds and
unknown-result treatment remain unchanged. Reported rental price is their
historical result, not an offer, authorization or expected cost for our work.
Public artifacts are not yet a self-contained exact-match experimental package
according to AppendixD; author-reported claims remain externally unreplicated here.

Mechanisms worth controlled study, after current data/training controls:

- Native search inner loop, independent-game inference batches, subtree reuse;
  Python orchestration outside leaf hot paths. Measure whole-loop cadence and
  target quality before a rewrite; GPU TensorRT/QAT cannot be assumed on4CPU.
- Broad restarts with full reconstructed history and untried branches, separate
  from replay resampling. Current real-PGN teacher roots broaden supervised
  coverage; they are not this paper's online self-play archive or self-learning.
- Replay window600k→20M, reuse credits4presentations/newadmittedrow, uniform30%
  plus capped policy-surprise sampling70%. These constants are not adopted;
  our compute, data size and learner/actor balance differ.
- Full searched endgame admission; actual cutoff search separated from observed
  terminal outcomes. Future bootstrap targets need provenance/calibration, not
  unknown caps silently converted to draw. Next-policy/remaining-length labels
  require correct perspective and known future masks.
- Function-preserving growth and paired strength qualification. Our earlier
  width transfer already used active random extra features with zero readout;
  failed results must not be re-diagnosed as merely dead features.

Especially useful negative evidence from sections4/6/AppendixC:

1. Predicted budget allocator improved deep-policy divergence, approximately
   1.18times uniform target fidelity at0.967times spend, but online training
   trailed by about60–100ladderElo. Learned difficulty/entropy controller fit is
   not proof of better learning. Our optional later decision-model idea needs a
   uniform-budget baseline and final strength, not merely surrogate accuracy.
2. An in-search stopper removed14%nominalsimulations but cadence improved only3%
   under actor/trainer overlap; paired strength effects unresolved. End-to-end
   critical-path work matters more than forward utilization or saved calls.
3. Fast/full search could discard useful chess policy targets and create batch
   tails. Go's playout randomization does not automatically improve chess.
4. Exact encoded-input cache hit0.970%; disabling sped the observed actor workload
   0.88%. Profile real repeats/overhead before building a large cache. A graph
   identity must preserve history/repetition/draw rules, not just piece placement.
5. A loss-based capacity promotion picked a roughly270Elo-weaker candidate after
   unequal catch-up exposure. Our data controls use identical common old/new
   holdout positions and separate, preregistered paired strength/speed gates.
6. TensorRT accepted refits while legal policy diverged badly. Serving success
   is not prediction fidelity; actual selected-model backend parity remains a
   requirement. CPU/MLX tests do not certify unavailable AppleMetal/CUDA devices.

Local convolution with every-second-block mean/max pooling was retained over
attention on serving cost, despite small fit differences. The reported
pooling-versus-SE gain is qualitative without preserved comparison results;
do not turn it into a measured strength improvement. Their from-to policy
mechanism overlaps our existing pairwise head; a rename or copied recipe would
not be novelty. 52/1880encoding differs from our104history/4672contract. No
unversioned action/encoding migration is introduced into existing recordings.

## Searchless RL beyond imitation, August2026

Szymon Miłosz, Piotr Duch, Szymon Grabowski,
[Beyond Search-Imitation: Prior-Directed Exploration for Searchless Chess](https://arxiv.org/html/2608.27757v1),
arXiv2608.27757v1/27August2026.
HTMLSHA5a7bfc6283b4be1b89b13177cd4c2df36a03450fc48b1f804ccf9b6c9f1e19d2.

Released strong Chessformer initialization, fixed base-policy prior, forward
mass-covering KL(base||student), value-outcome entropy sampling temperature,
single gradient update/freshbatch, truncated importance weighting and TD(0)
with EMA value target. Reported2000steps×1024parallel-game transitions, matched
regularizer controls;100kpuzzle accuracy93.9→94.9%, mate-in-four77→81%. The
reported searchless strength improvement is modest; puzzles and play can diverge.
Evaluation includes200games perround-robinpair and1000againstbase; reported
rating depends on that pool. We did not reproduce the network or results.

Forward KL is soft prior cross-entropy minus constant prior entropy. It preserves
mass on plausible alternatives; reverseKL is mode-seeking. This supplies a
concrete later self-play ablation: unanchored versus fixed-prior anchored updates
at equal fresh-data/update/search/wall budgets, with own-outcome feedback and
independent games. It is not the same as only imitating new MCTS policies.
Current teacher-pretraining protocol adds no such term. Weak/miscalibrated
HarbiChess value predictions do not justify automatic entropy temperature, and
a poor prior can preserve poor moves. No teacher-surpassing guarantee follows.
This strengthens the motivation for quality provenance in future decision/LLM
datasets, without starting a second model project or mandatory Laya integration.

## Particle MCTS, May2026 revision3

Yaniv Oren, Viliam Vadocz, Joery A.deVries, Wendelin Böhmer, Matthijs T.J.Spaan,
Hendrik Baier, [PMCTS: Principled Parallelized Inference Time Scaling with Particle
Monte Carlo Tree Search](https://arxiv.org/html/2605.08982v3).
HTMLSHAbbe9475fdd505a9ccc0d3f4f945e583acb23a16c14587a8e2553037e4574e3bc.

Stochastic independent particle trajectories, batch expansion and duplicate
accounting address deterministic repeated selections/root parallel overlap.
The paper's improvement argument assumes unbiased, uncorrelated evaluations;
our previously demonstrated inaccurate value estimates do not satisfy that
assumption by declaration. GPU batch benefits are not CPU wall-clock benefits.
Possible later matched-total-wall serial-versus-batched-search control, after a
reliable evaluator, records duplicate work and legal/history fidelity. No
parallel-search replacement or extra simulation budget is introduced now.

## What this changes now

Keep running registered broad-v4collection unchanged. Complete every-row audit,
unchanged-source merge, lossless packed preparation, common old/new cold baseline,
then freeze paired training controls BEFORE updates. Separate actual Stockfish/
neural matched games and speed from teacher fit. Preserve all failed arms,
unknown caps and full native resume. Only afterwards choose the next controlled
self-play/representation/search hypothesis from measured bottlenecks.

The close September system study already covers adaptive allocation, replay,
restarts, from-to heads, global pooling, size growth and whole-loop efficiency;
the August study covers prior-directed RL exploration. A future paper needs a
demonstrated contribution beyond these, independent seeds/strength/latency
ablations and a reviewable evidence package. We have not established such novelty
or main target strength yet. Article ambition does not relabel failed experiments.
