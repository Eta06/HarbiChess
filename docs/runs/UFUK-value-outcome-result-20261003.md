# UFUK: controlled late-outcome value contribution result

The registered played-strength gate **FAILED**.
The500update candidate failed native QMAE retention and stopped; that failure
remains failed. The earlier250checkpoint passed the original preregistered
minimum-heldout-CE-among-valid-checkpoints rule. This is native-anchor plus
observed-self outcome value learning, not pure self-play from scratch. Main
strong/fast/reliable self-learning Stockfish/AlphaZero objective is still unmet;
no general Elo, teacher-exceeding result or novel RL mechanism claimed.

[Original protocol](UFUK-value-calibration-preregistration-20261003.md),
[preserved publication failure/full-state reporting migration](UFUK-value-reporting-v2-20261003.md),
[explicit after-measurement qualification-code correction](UFUK-value-checkpoint-qualification-20261003.md).
The driver mistakenly added a whole-run-completed veto absent from the written
protocol. Original false gate/failed500state preserved; separate qualification
audit applies the previously committed checkpoint rule, with no changed threshold,
seed, extra training or candidate selection after matches.

Both arms initial native18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae,
resetoptimizers. Only31,311value-specific parameters trained,41,186shared/policy
parameters bitwise frozen at every saved checkpoint; full policy logits unchanged.
Native32row uniform plus own32row game-balanced sampling; control loss
0.5*nativeCE+0*selfCE, candidate0.5*nativeCE+0.5*selfCE. Same500actual sample
sequences reconstructed fromseed20261016, same common-checkpoint TorchRNG.
CPU1/FP32/AdamW5e-5/wd1e-4/clip5, max4000/eval250/patience1000.

Cached inputs:10000existing native training reference rows from36allowed families,
2663own late terminal rows/84games,704heldout rows/22games/11families; no unknown
cap targets. Final32preterminal states per known game, entire12family holdout
defined before selection;0current-position train/heldout overlap. Native anchor
excludes heldout families/currentpositions. Initial model previously saw these
teacher training families; data-generating policy actors used rolling self data.
This is new-stage gradient separation, not retrospectively independent from-
scratch collection. Native20k observer is never trained on; known-terminal/late
selection is not representative of all game states.

| Arm | Step | Own heldout gameCE | Native20k valueCE | NativeQMAE | Retention |
| --- | --- | --- | --- | --- | --- |
|anchor|0|1.326842336|0.654349148|0.336882945|True|
|anchor|250|1.334927079|0.654474388|0.335354250|True|
|anchor|500|1.365504809|0.656267123|0.334951198|True|
|anchor|750|1.359597345|0.667796461|0.340172644|True|
|anchor|1000|1.368629291|0.656346125|0.336069480|True|
|self|0|1.326842336|0.654349148|0.336882945|True|
|self|250|1.063226538|0.667610130|0.349700419|True|
|self|500|0.983875134|0.683131530|0.358984565|False|

Control best0/1000patience stop; candidate selected250 heldout gain/controlgap
0.263615799>=0.10/0.05. Candidate250 nativeCEdelta0.013260981<=0.03,
QMAEdelta0.012817474<=0.02. At500QMAE0.358984565 exceeds0.356882945;
no500promotion or continuation. Loss qualification is not game strength.

Actual source66a5e1bd21e4b90792f96eb34aa43c98442369ed control training250→1000
full native fresh-process resume. Computation reached stop, then immutable result
publication failed(exit1); receipts/checkpoints/partial result preserved.
Reportingv2 source13cfd890a2f33e57fb51fe20229ad8173145cc28 migrated complete
control0/1000 tensors,optimizer,RNG/cursor/cache, preserving originals and exact
sample traces. Both complete states compared bitwise, zero extra control updates.
Candidate250→500 fresh-process full resume at samev2 source; stopped on retention.
Source difference is publication/profile-only, not an unrecorded numerical change.

Shared preparation outerwall58.197758s (charged to eacharm). Control total
121.629169s includes originalfailed
publication plus migration/republication; candidate
94.044871s. Both<1800s;
native observer loading included in train timers. Different stop lengths are not
equal-workload throughput. Existing4CPUquota/16GiB/noGPU/no paid resources.

Selected candidate250 SHA25b97b4d7388f4ef97abba6bc0506ed921af9e416dc4c47271333eb08e3e9392; anchor0 SHA
bfaddba71d6ef96ca29be51ea2e2fa8b3c53095f0edde8899b88fe5d430e8c1a. Anchor0 tensors equal original18f; provenance
metadata changes model file hashes through full-state migration. Actual trained
candidate inference matches realMLXCPU full logits/WDL to2e-5; native checkpoints
are Torch optimizer states, portable MLX loading remains weights-only. AppleMetal/
CUDA tests unavailable. Latest508tests/zero skips43.94s; changed-file Ruff passed.
Historical scalar value code restored after filename-collision preflight; failed
collection/output attempts retained, no old code/results silently removed.

Fresh24development roots/32plies bookSHA452d8fd829918893ef08530957cea70cf1b4e6f40ca64a2dcda6eac23a99b1c6,
frozen before first update, uniformlegal fourply extension, no engine/model filter;
all80511native labels/24prior arena histories excluded at current roots. Same
development families reused; no independent Elo or pooled prior samples.
Both16sim/max4/G0/value_scale0.1/maxvisit_init50/CPU1/eager/batch1,
48colourmatched games/arm/seed20261016/max240plies/claim_draw=True/900s perarm.

| Arm | Wins/actual draws/losses/unknown | Score | 24pair bootstrap95% | Arena wall | NN calls |
| --- | --- | --- | --- | --- | --- | --- |
|candidate-vs-anchor|11/17/17/3|0.437500|[0.375, 0.5]|181.687720|84816|
|candidate-vs-stockfish32|9/12/27/0|0.312500|[0.21875, 0.40625]|42.556141|18950|
|anchor-vs-stockfish32|8/14/25/1|0.322917|[0.22916666666666666, 0.4166666666666667]|46.706823|20382|

Primary conservative Hoeffding95%[0.16027868896125547, 0.7147213110387445]; bootstrap and
conservative bound assume independent opening families, do not account for all
earlier exploratory model/method selection. Caps score0.5 only diagnostic and
remain unknown. Registered score>0.60/lower>0.50/caps<=0.10 unchanged.
All144games/14417plies legally replayed and durable
traces matched. SF candidate-minus-anchor pairedscore
-0.010416667,95%[-0.10416666666666667, 0.08333333333333333].
SF19/Threads1/Hash16MiB/32requestednodes; actualnodes recorded per move, neural
simulations/engine nodes are unequal compute. Different game lengths prevent
an isolated speed ratio; these remain weak-budget references, not strong engines.

[Exact evidence](UFUK-value-outcome-evidence-20261003.json) preserves inputs
metadata/source hashes, both reporting versions/failure, complete sample traces,
checkpoint manifests, qualifications and every game/engine node count. Binary
cache/model/optimizer/replay preserved separately; remote Release upload still
400BadContentLength, so new remote binary backup is incomplete. No older failure
or failed500checkpoint is rewritten as successful self-learning.
