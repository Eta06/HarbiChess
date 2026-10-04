# UFUK: sparse current-position value with matched old-value training

Before implementation/transfer/production updates. Context native learning and
classical tactical controls remain failed. Hypothesis: a compact piece-square
value MLP can fit independent engine WDL labels better and remove the expensive
value CNN/tower from CPU inference. It may fail; no result or novelty assumed.
The separate CPU research read Stockfish NNUE trainer
13b44569e0674fe7267122396ea7bd0946f10115/model/nnue.py (12679bytes,
SHA18311ca43be899e1026a962e2d4a207e774334874a07befa194fbe4898bf7d8b),
model/model.py, features/halfka_v2_hm.py and docs/features.md. Actual training
uses sparse king-bucket/perspective features, quantization/layer stacks and a
scaled score/outcome blended power loss. We do not copy that trainer, features,
weights or Stockfish tuned constants; standard sparse/MLP math is not original.

Original18f pairwise policy/shared parameters retained bitwise. Optional
value_sparse schema1/channels128/hidden64, only allowed with canonical104-channel
pairwise model. Feature input832 = current64x12piece planes plus full64square
en-passant plane. Add an8metadata->128linear projection to832->128feature
projection; clip to[0,1],128->64clip[0,1],64->3STM W,D,L logits. Current-position
head intentionally excludes old piece-history planes; repetition/rule50/STM/
castling metadata and full EP square retained, exact history-sensitive terminal
rules stay in search. Old policy still uses full104history encoding. Sparse
active-column sum is mathematically equivalent to dense projection; no claim of
incremental caching/quantized speed until implemented and actually tested.

Conversion version1 is explicitly policy-preserving/NEW RANDOM VALUE/optimizer
reset. Old value parameters stay archived/registered but inactive in candidate
forward. This is not initial value identity or full resume. Old specifications,
checkpoint/replay formats remain strict; unsupported old conversions refuse
new-head models. Within each backend full native optimizer/RNG/sampler resume
is separate from portable weight transfer. No cross-backend optimizer equality.
Real Torch/MLXCPU full/masked loss/gradient/update/portable and frozen-policy
tests, unknown-schema checks, source/callerTorchRNG preservation and exact native
split-update resume before production. AppleMetal/CUDA unavailable, no skips.

Control: original18f, shared/policy frozen, train only the active original value
modules (value_*,invariant_value_*,global_value_*). Candidate: onlyvalue_sparse_head.*,
all original72497parameters frozen. Keep native80511dataset/48train16validation
whole-family split/caps40k20k, seed20261022. CPU1FP32/AdamW3e-4/wd1e-4/clip5/B64,
max12000updates/eval500/patience2500/1800whole-wall per arm including inputsetup
and500pause/fresh-process fullresume. Same sampled indices at common steps;
unchanged oracle losses/sampler/optimizer, no native teacher target edits.
Stop nonfinite/frozen mismatch/source/runtime/input mismatch/budget/patience.
Audit frozen parameter hashes at pause and every evaluation/publication.

Select minimum native valueCE among valid checkpoints with exact unchanged
policy. Require selected candidate valueCE <= original18f initial valueCE−0.05
AND <=selected control valueCE−0.03, and candidate QMAE <= original initial−0.02
AND <=selected control−0.01. Initial candidate random-value loss is not the
qualification reference; easy improvement from random initialization cannot
qualify. Unchanged policy means original totalCE patience matches valueCE order.
Legacy core default both-heads qualification is not this experiment's gate.
No arena if this gate fails; no lucky update/budget extensions.

Before updates freeze24new12ply uniformlegal stage stress families seed22 with
the existing complete first4history/FEN/native/prior-arena exclusions, including
unused context/tactical books. No model/engine quality filter; not natural-opening
Elo or exhaustive inherited pretraining holdout. Fixed18real full-history latency
probes from the width stage,CPU1/eagerlegalmaskedB1,10warmup200alternating rounds,
no concurrent training; full policy+value tasks matched, median candidate/control
ratio<=1.05. A separate sparse/value-only microbenchmark cannot replace game cost.

If native qualified, candidate-vs-selected-control and each-vs-SF512,48colorpaired
games/arm,new24families,16sim/max4/G0/value_scale0.1/maxvisit_init50/max400plies,
CPU1B1/seed22/900whole-wall perarm. SF19Threads1Hash16MiB512requested+actualnodes.
Strength gate score>0.60,24pair95%bootstraplower>0.50,caps<=0.10; conservative
Hoeffding and paired SF delta uncertainty, source/checkpoints/trace/NNcalls and
wall/median/p95move times recorded. Unequal simulation/engine-node compute,
no general Elo or Stockfish/AlphaZero equivalence. Even bounded passes would
need a separate fresh closed-loop self-learning protocol, not imitation-success
or automatic teacher superiority. Existing4CPU/16GiB/noGPU only, no paid resources.
