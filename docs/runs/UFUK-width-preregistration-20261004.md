# UFUK: controlled width after measured large native excess loss

Before new model transfer/update. Native20k target entropy does not explain
policyCE:1.151859132H versus2.632099348CE, excess1.480240216; WDL excess
0.520756475. Large reducible loss motivates a new capacity/optimization test,
not certainty that architecture is the bottleneck. Previous depth/adapter and
policy/value self-learning game failures remain failed.

Hypothesis: wider shared spatial features improve heldout native policy and
playing strength under bounded CPU latency. Pairwise head/invariant feature
semantics/rules/104encoder/4672actions/STM WDL/alternating search unchanged.
Conventional function-preserving widening is not a novel algorithm or paper claim.

Both original18f and widened transfer begin mathematically the same function.
Shared trunk16→64, independent value tower16→32, same2resblocks/tower and hidden
dimensions. Old channels remain exact parameter subblocks; new channels have
random active features but zero connections into inherited outputs/heads,
avoiding dead new channels. Pair-hidden global feature column offsets and pooled
value mean/max offsets remapped explicitly. Allow2e-5 full/masked initial-logit
numerical tolerance due changed CPU kernel/reduction geometry, not a bitwise
initial-function claim. Initial metrics recorded for both arms. Seed20261017,
caller RNG preserved, old weights immutable; optimizer reset/weights-only width
transfer schema1, then full native within-arm optimizer/RNG/data resume.

Real Torch/full/masked and MLXCPU portable outputs, trainable new features,
strict dimensions and original-file integrity must pass before training. Apple
Metal/CUDA unavailable; no tests skipped to hide this. Old architectures/replays/
checkpoints retain loaders; version1 specification widths already explicit.

Native80511 dataset, max40000train/20000validation rows, seed20261017, original
whole48/16family split; no newengine query/own observed value label mixed in.
Both allparams/CPU1/FP32/AdamW5e-5/wd1e-4/clip5/batch64, max6000updates,
eval250/patience1500,1800s whole perarm with input preparation conservatively
charged. Same sampled-index/TorchRNG traces at common steps, pause250/fullresume
in fresh processes. Validation is reused development data, not independent Elo.

Select minimum native policyCE among checkpoints whose valueCE<=initial+0.02
and QMAE<=initial+0.02. Require width policyCE gain>=0.05 versus initial and
>=0.03 versus similarly selected native control; valid selected checkpoint,
finite/integrity required. Stop nonfinite/retention/source/hash mismatch, budgets
or registeredpatience; preserve invalid/later failures, no lucky extension.
No game arena if selected-checkpoint qualification fails.

Before firstupdate freeze24new36ply roots by fouruniformlegal suffix moves from
the32ply value arena book, seed20261017, exclude currentroots from native labels
and prior complete arenas; no model/engine filter. Same development families.

Ifqualified, widthvscontrol and eachvsSF32,48colour-paired games/arm,
16sim/max4/Gumbel0/value_scale0.1/CPU1/eager/batch1/max240/900s perarm/seed20261017.
SF19/Threads1/Hash16MiB/32requested and actualnodes. Primaryscore>0.60,
bootstrap24pair95%lower>0.50,caps<=0.10. Same sims are unequal compute at different
widths; no equal-compute superiority claim. Report complete arena wall, NNcalls,
permove median/p95 timing, actual engine work and pairedSF uncertainty.

Separately fixed18real FENs/eagerlegalmasked B1, warmup then200rounds/fixedseed,
alternating model order; record setup excluded and inference-only latency.
Speed qualification width/control median latency<=2.0 and finite numerical
outputs. Passing game strength without speed gate is not strong-and-fast success;
even both passes remain bounded development evidence, not Stockfish/AlphaZero
level, general Elo, reliable closed-loop self-learning or publishable novelty.
Existing4CPUquota/16GiB/noGPU/no paid allocation.
