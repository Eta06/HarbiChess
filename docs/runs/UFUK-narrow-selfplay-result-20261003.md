# UFUK: six-generation narrow-search self-learning result

The preregistered final-versus-initial model strength gate **FAILED**.
Both models use identical16simulation/max4/Gumbel0 search; this comparison
does not count the prior fixed-model root-width improvement as learned strength.
The main strong/fast/self-learning Stockfish/AlphaZero-level objective is unmet.
No universal Elo, teacher-exceeding result or publishable originality is claimed.

[Preregistration](UFUK-narrow-selfplay-preregistration-20261003.md), fixed clean
training/arena source96c4ef48485749cf0e58f117ea3867fce726686a. Original native
weights18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae;
final generation6 SHAf038e7a7d30de5865c0c9aec0fd21c41cbdd57d2a14c88057c9fb8391d4b79bb.
Weights-only initialization/resetoptimizer followed by five fresh-process full
optimizer/RNG/replay/cursor resumes. Final checkpoint actually loaded separately:
step768,nextgame192,10optimizer state entries,CPU1/Torch2.14.1+cpu/deterministic.

Sixgenerations×32games=192own-search games, 35092positions,
106terminal games/15140knownoutcome rows,
86unknowncap games masked. No teacher labels,
engine actors or validation gradients during this learning. Pairpolicy16,914
parameters trained;55,583shared/value parameters bitwise unchanged at all seven
checkpoints. Native20k valueCE0.654349148/QMAE0.336882945 remain exactly unchanged;
this is policy-only learning, not value learning.

| Generation | Rows | Terminal games | Known rows | Rolling heldout known | Native policyCE | Retention |
| --- | --- | --- | --- | --- | --- | --- |
|1|6131|15|2187|0|2.644139828|pass|
|2|5857|18|2609|0|2.652822158|pass|
|3|5645|17|2165|113|2.663263867|pass|
|4|5854|22|3534|228|2.678850748|pass|
|5|5520|18|2272|375|2.690413188|pass|
|6|6085|16|2373|262|2.705301860|pass|

Native policyCE2.632099348→2.705301860
(retention tolerance+0.15); allsix gates passed. Early heldout WDLloss0 with
zero terminal support represents absent supervision, not improvement. Every
shard validates version/checksum/legality, all192 full histories/terminal targets
and source checkpoint hashes audited. Arena current-root contamination absent;
development family reuse and prior model selection limit generalization.

Auditv1 was interrupted after624s because its transition loop rebuilt a fresh
rules cache for every row; this was an inefficient audit, not a training failure.
Its source/log/interruption receipt are retained. Auditv2 replays one incremental
board per complete game with identical chronology/legality/outcome/holdout checks;
versioned shard validation and all checkpoint hashes still run in full.

Actors64sim/max4/Gumbel1, fourspawnCPU1 workers; learnerCPU1/FP32/batch64,
AdamW5e-5/wd1e-4/clip5/seed20261015,128updates/generation,rolling3generation
game-balanced sampling.768updates/49,152sampled rows. Initialactor openings are
48eight-ply training families. Versus the older wide self experiment, trajectories,
generations and root width allchanged: no single-variable causal claim against it.

Actual self-play wall 1728.853959s/2,241,003NN evaluations,
1296.236NNpositions per selfplay wallsecond,
20.298playedpositions per selfplay wallsecond;
training update wall 19.112051s. Sum fresh-process command outerwalls
2150.424768s includes process setup/within-loop input preparation.
Driver reports2176.131148s after initial nativeobservercache loading; that initial
cache loading is excluded, so this is not a complete startup-inclusive timer.
Existing4CPUquota/16GiB/noGPU/no paid resources. Summed worker CPU is distinct
from parent CPU/outerwall; no GPUutilization or isolated model speed claim.

Fresh24roots/28ply bookSHAeae05e0080703e5571c4db689d5a165e39d030515f1b294515e82132d4810d0c,
seed20261015; uniformlegal suffixes frozen before training, no engine/model filter.
All80511native labels/21prior arena files excluded at current roots. Same24
development families reused, not independent confirmatory Elo.48colour-matched
games/arm,16sim/G0/max4/value_scale0.1/maxvisit_init50/CPU1/eager/batch1,
max240plies/claim_draw=True/900s perarm. Caps count0.5 diagnostically and remain
explicit; gate score>0.60/bootstrap lower>0.50/capfraction<=0.10, no lucky extension.

| Arm | Wins/actual draws/losses/unknown | Score | 24pair bootstrap95% | Arena wall | NN calls |
| --- | --- | --- | --- | --- | --- |
|final-vs-initial|14/15/10/9|0.541667|[0.46875, 0.6145833333333334]|207.656135|94650|
|final-vs-stockfish32|8/17/23/0|0.343750|[0.25, 0.4479166666666667]|52.767839|22816|
|initial-vs-stockfish32|9/17/22/0|0.364583|[0.2708333333333333, 0.4583333333333333]|47.474246|21194|

All144arena games/14999plies legally replayed;
durable per-move traces match completed histories. Primary conservative
Hoeffding95% bound[0.2644453556279221, 0.8188879777054112] is reported beside the bootstrap;
both assume independent opening families, without accounting for prior exploratory
model/method selection. PairedSF final-minus-initial
score-0.020833333, bootstrap95%
[-0.11458333333333333, 0.08333333333333333]. SF19/Threads1/Hash16MiB/32requestednodes;
actual node overshoot recorded per move. Neural simulation and engine node
budgets are unequal compute; different lengths are not an equal-workload
throughput comparison. No Stockfish-level strength or newarchitecture novelty.

FinalSF arm actualnodes61349 across
1372engine moves, mean44.715015,
range32..305; initialSF arm
actualnodes58965 across1274,
mean46.283359,range32..147.

[Exact evidence](UFUK-narrow-selfplay-evidence-20261003.json) preserves every
command, generation session, full-history arena trace, gates/audits, source
generator, tests and observer cache metadata. Native checkpoints/model/replay
binaries stay separate; exact text evidence is not a binary backup. All prior
wide-search/depth/adapter/selector failures remain failed. No model promotion
unless the registered model gate passes; even a pass is bounded development
policy self-learning, not the main objective.
