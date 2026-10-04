# UFUK: global policy context after local capacity failures

Before new transfer/production updates. Native width failed, Qrange floors
failed, offline8000own-policy model failed its new stage family game gate.
These results remain failed; repeating the same updates is not justified by CE.

Hypothesis: two small global64square self-attention blocks in the policy path
model distant piece interactions better than the inherited local features/head.
This does not assert attention is necessary or will fix learning. Re-read
[Lc0 primary attention forward](https://github.com/LeelaChessZero/lczero-training/blob/7c5d756ea6bb3531fb14a9b4df231577b1aa1081/tf/tfprocess.py#L1264):
QK/sqrt(depth), softmax across64squares, value aggregation, feedforward,
optional Smolgen biases and DeepNorm. Source75453bytes SHA
8a20692dbe2c684e89bc9c8fb41abb83578f069079bdd89ebcf163079337d4fc.
We independently implement ordinary attention math, not import/copy Lc0's
trainer or claim its strength. No Smolgen/DeepNorm/GPU-size scaling copied here.
Standard attention/zero residual initialization is not a novel algorithm.

Original18f pairwise16channel trunk. Optional policy_context schema1,
blocks2/heads4, tokens64×16; LayerNorm epsilon1e-5 inside each residual branch,
QKV projection, scaled-dot-product attention, output projection, ReLU feedforward
16→32→16 and learned64×16zero-initialized positional features. Attention/output
and secondFF weights+bias zero so initial function is preserved; active random
QKV/firstFF features allow bridges then features to learn. Context applies only
to policy tokens, never mutates original trunk or value. Original55583shared/
value parameters frozen, onlypair_* and policy_context_blocks.* trainable;
both oldplain models and oldlocal adapters/replays retain strict loaders.

Version1 optional specification and explicit policy-context-v1 weights-only
conversion/resetoptimizer; no optimizer transplant or cross-backend full resume.
Before updates verify real Torch/full/masked and realMLXCPU loss/grad/update,
initial identity tolerance2e-5, original file/callerTorchRNG preserved, exact
frozen value and full within-Torch split-update native resume. Unknown context
schemas/dimensions fail loudly. AppleMetal/CUDA unavailable, no skip workaround.

Both original and context fresh optimizers, native80511dataset, caps40000train/
20000validation, seed20261020, same48/16whole-family development split.
CPU1FP32/AdamW1e-4/wd1e-4/clip5/B64/max8000/interval250/patience1500,
1800s whole per arm including input setup/fresh-process pause250+fullresume.
Control onlypair_*; context pair_*+context blocks. Native value is constant,
so totalCEpatience and policyCEordering agree. Audit frozen hashes at250boundary
before continuing and at each published checkpoint; mismatch invalidates/stops
the run and preserves latest complete safe state. Stop nonfinite/source/hash/
budget/patience mismatch; no lucky extensions. Same sampled indices/RNG at common
steps. Legacy core default qualification is not this experiment's gate.

Select minimum policyCE checkpoint whose frozen-state hashes/value outputs stay
exact. Require context policyCEgain>=0.05 frominitial and >=0.03 versus selected
matched native control. No arena if failed. Lower native loss is supervised
representation evidence, not yet self-learning or strength.

Before updates freeze24new12ply uniformlegal stress families with seed20 using
the same prefix-history/FEN/native/prior-arena exclusion method as the retained
offline-control stage, now including that book/144games. No engine/model filter;
new stage prefixes do not establish complete inherited-pretraining independence
or natural-opening Elo. Currentroots excluded from80511native labels/priorarenas.

Fixed18real full-history latency probes already frozen before width updates,
eagerlegalmasked B1/CPU1,10warmup then200alternating-order rounds; no concurrent
training. Inference-only context/control median ratio<=1.5; finite outputs,
setup/encoding/search excluded and separately reported. A strength pass with
failed speed does not establish strong-and-fast success.

If lossqualified, context-vs-control and each-vs-SF512,48colourpaired games/arm,
16sim/max4/G0/value_scale0.1/maxvisit_init50/max400plies/seed20/900s whole perarm,
eagerCPU1B1. SF19/Threads1/Hash16MiB/512requested plus actualnodes. Primarygate
score>0.60/24pair95%bootstraplower>0.50/caps<=0.10; conservative Hoeffding bound
also, pairedSF uncertainty, durable legal traces, wall/NNcalls/median+p95 move
times. Unequal simulations/engine-node compute, no general engine Elo.
Even passes remain bounded native-learning/development games, not fresh
closed-loop self-learning, Stockfish/AlphaZero strength or original invention.
Existing4CPU/16GiB/noGPU only, no paid allocation. Dataset/LLM/Laya work deferred.
