# UFUK: all-legal CPU search versus learned-policy pruning

Written before implementation or diagnostic measurements. Sparse-value native
learning remains FAILED; its selected1500checkpoint is an experimental evaluator,
not a promoted model. Main strong/fast/self-learning objective remains unmet.

Hypothesis: at the same per-position wall budget as existing16simulation/max4
Gumbel MCTS, a bounded iterative all-legal alpha-beta search can recover moves
excluded by weak policy priors. Compare material-only and sparse-learned value
search to separate classical search benefit from learned evaluation benefit.
Ordinary alpha-beta/quiescence/sparse column sums are not novel inventions.
No Stockfish source, weights, tuned tables or new external dataset is copied.

Implement an optional Torch/MLX-free NumPy evaluator for existing versioned
sparse-value-v1 weights. Current piece-square columns and full EP square plus
exact8metadata must match the104encoder-derived head/STM W,D,L/Q. Read-only
weight snapshots, finite/shape/schema validation, no parameter/cache mutation.
Do not infer terminal rules from value; search checks exact outcomes with full
history/claim_draw. No FEN-only transposition cache. Preserve old model formats,
training resume and both existing backends. Test real trained Torch/MLXCPU parity
on special moves/black orientation/repetition/rights/rule50 and history restoration.

Implement experiment-only iterative negamax alpha-beta with all legal root
moves; capture/promotion/check-evasion quiescence depth6, ordinary MVV/LVA
ordering with prior-iteration best root first. No neural root pruning, null-move,
late-move reductions or transposition table. Completed depths only; timeout or
node limit aborts the entire current iteration, never publishes a partially
searched root as an exact minimax result. Legal deterministic fallback if no
depth completed must be explicitly counted. Quiescent depth truncation while
in check aborts the iteration instead of pretending checked stand pat is legal.
Outside-check stand pat is permitted. Per-search node cap100000/maxdepth6;
check wall before every visited node/evaluation. Report finite-unit wall overshoot,
completed depth, all root moves visited at completed depths, normal/qnodes,
evaluation calls, check-depth aborts/time/node aborts and legal traces.

Diagnostic: reuse immutable128native positions/16families/all3433independent
SF32768legal-move references; no new engine queries or optimizer updates.
Original18f weights and prior MCTS16sim/max4/G0/value_scale0.1/maxvisit_init50,
CPU1FP32/eagerB1/seed20261023. Warm existing evaluator on fixed18history probes.
For each position measure baseline MCTS once first, then give EACH alpha-beta
arm that measured complete baseline wall as its search budget. Alternate order
of material/learned arms by position. Baseline-first ordering is a limitation;
record actual arm wall/ratios, no concurrent training/benchmarking, root setup
and board history construction counted in every arm. Under1000whole-wall seconds
and3500baseline neural calls; source/model/reference/probe/script SHA receipts.

Sparse selected checkpoint SHA
15882e0e9c967f6231aefa0a7121cef8daedd65895b9ee0cb916ac8cab4c93ed;
original18f SHA18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae.
References SHA6e65d3d0f4d6c507deaae87ac7676fa1707f43914dbec832b930537b0256db50;
latency/history probes SHA1089fd0cca308c24bb040a840352d0ad18aa85a7bdcd17608133456bf08bed9f.

Mechanism quality gate: learned arm selected-reference expected-score mean
gain>=0.01 AND family-cluster bootstrap Bonferroni2contrast97.5% lower>0 versus
BOTH MCTS and material arm (10000replicates/seed23,16families). Record nominal95%
and conservative Hoeffding intervals too; development-family conditional evidence,
not independent Elo. Median actual learned/MCTS whole-root wall<=1.05, <=2%
roots with no completed depth, no illegal move/nonfinite/history mutation/budget
breach. The material comparison is also reported but cannot prove model learning.
If mechanism gate fails, no game arena/promotion/lucky knob or budget extension.

If qualified, freeze a NEW24family12ply uniformlegal stress book before games,
with existing full prefix/history/FEN/native/prior-arena exclusions and no quality
filter. Separate prospective game protocol then registers matched whole-move
wall budget for learned/material/MCTS and SF references, colour/opening pairings,
strength/cap gates/uncertainty. A diagnostic pass alone is not chess strength or
self-learning; any subsequent closed-loop training needs a fresh protocol and
same-search initial/final matches. Existing4CPUquota/16GiB/noGPU only, no paid compute.
