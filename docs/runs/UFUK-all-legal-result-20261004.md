# UFUK: matched-wall all-legal search result

Mechanism **FAILED**. No arena, promotion, default-search change or lucky
budget/knob extension. Main strong/fast/reliable self-learning goal remains unmet.
[Protocol before implementation](UFUK-all-legal-preregistration-20261004.md).
Sparse-value native training remains failed; its1500checkpoint was only an
experimental evaluator, not a replacement for original18f.

Clean pinned source `560f5f8f6b84b91ccb56438438f6a604ff2512de`. Framework-free NumPy active-column
sum reproduces sparse-value-v1 STM W,D,L on current pieces/full EP square and
8metadata, strict FP32/schema/shape/finite tensors, read-only weight snapshots.
Actual selected trained18history probes: NumPy/Torch maxlogit error
4.17232513e-07, NumPy/realMLXCPU
9.53674316e-07. Before diagnostic544tests passed,
zero skips45.61s, targeted6passed2.21s, Ruff passed; actual Torch/MLX updates,
special moves/metadata/rules/time/node/checked-depth abort tested. AppleMetal/CUDA
unavailable, untested. No optimizer migration or incremental accumulator claim.

Ordinary experiment-only iterative all-legal negamax alpha-beta/maxdepth6,
quiescence6/MVV-LVA/100000total-node cap, no TT/policy pruning. Exact full-history
outcome/claimdraw; checked qdepth truncation aborts entire iteration, never stand
pat in check. Root publishes only whole completed depths, explicit legal fallback
with valueNone if no depth completed; finite-unit wall overshoot measured. No
Stockfish GPL source/tuned tables/weights copied, no novelty assumed.

128positions/16development families/all3433independent SF32768move references
reused, no new teacher queries or gradients. MCTS16sim/max4/G0/value_scale0.1/
maxvisit_init50/CPU1/eagerB1/seed20261023 measured first per position; EACH
material/learned root received that actual whole-root wall budget including
board reconstruction. Material/learned order alternated. Baseline-first temporal
bias, finite scheduling overshoot and development-family reuse are limitations.
Baseline exact selected/visits/policy matched prior immutable root-budget control;
reproduction is not fresh statistical evidence.

Learned selected-reference score vs MCTS mean
**-0.087808594**, adjusted97.5%family bootstrap
`[-0.12737890625, -0.0474609375]`;
vs material **-0.061332031**, adjusted interval
`[-0.0928515625, -0.031195312500000006]`.
Both fail mean>=0.01/lower>0. Nominal/conservative Hoeffding intervals retained;
conditional bootstrap cannot establish general Elo or account for every prior
exploratory hypothesis. Negative mean and intervals are reported as failure.

Median actual learned/MCTS whole-root wall ratio
**1.002161464** <=1.05cost passed;
no completed depth on **18.75%** roots,
required<=2% failed. Learned 1987normal+
23182quiescent nodes,
21410value calls;
material 4054normal+
29266quiescent nodes,
26777value calls. Each abort/depth/overshoot
retained per root; no partial-root score claimed as exact minimax. These results
do not show that all-legal search or NNUE can never work; this weak evaluator,
Python rules, conservative truncation and finite wall combination failed.

Actual 2176NN calls including18warmup,
17.190999s whole diagnostic including startup/warmup, cap1000s.
All legal histories/actions/selected reference scores/counters/complete-root
coverage and both adjusted bootstrap contrasts independently recomputed.
Source/weights/reference/probe/method/script hashes preserved in exact evidence.
Original models/replays/data and default neural search preserved.

Existing4CPUquota/16GiB/noGPU/no paid resources. Source5files (preregistration+
4implementation/test files) actually downloaded from pushed immutable Git and
hashed. Author+committer Emir Tunahan Alim <emrtnhalim@gmail.com>, pushEta06/admin,
no coauthors/one meaningful file per commit. Space18important prior result and
next protocol read back exactly with72untouched blocks. Git UTF8 evidence is not
binary checkpoint backup; new Release uploads400BadContentLength remain blocked.

Next separate hypothesis: jointly adapt trunk/policy/value instead of freezing
the shared representation in the failed context/value experiments. It requires
a matched joint old-model control, prospective budget/gates and fresh same-search
games if qualified. More training by itself is not evidence of self-learning.
