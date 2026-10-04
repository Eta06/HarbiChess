# UFUK: retained offline own-target control strength failed

The new independently preregistered model strength gate **FAILED**. Offline
own-target head training8000updates did not demonstrate playing improvement.
The previous additional-adapter representation gate remains failed; neither
its control nor the adapter is promoted. Main strong/fast/reliable closed-loop
self-learning and Stockfish/AlphaZero-level goal remains unmet.
[New protocol](UFUK-offline-control-preregistration-20261004.md),
[old training/capacity result](UFUK-policy-adapter-result-20261003.md).

Original heldout-selected control8000only; no new checkpoint scan, gradients or
teacher labels. Old training source59dde8d/512000sampled own rows; candidate
SHAef2ee1d26bb9b0e80a9e849009c7ed326b43afe9a9881f0160ad2cb14a6dc96a,
initial18f. Lower ownCE0.076538and native-policy retention+0.112699 had passed;
this new arena tests their playing consequence, not retroactively changes the
old adapter comparison. Complete native checkpoint actually loaded at8000 with
10Adam states, RNG/cursor/bound data;55583inherited parameters bitwise identical
toinitial,16914policy trainable. Zeroextra optimizer updates in this test.

Clean arena sourcebbdac81feead83971304a8be728fa0cbddfcf061. Before games frozen24new12ply
uniformlegal stress families/seed20261019, bookSHA
34436236462ca16d6802a3b85398ef51a4eac2d1bb6c849e70e874276a2b8bac.
1202available input/book/arena files hashed;121existing first4histories excluded
alongside their4ply FEN keys (transpositions not independent), plus currentroot
exclusion from80511native labels/27prior complete arena files. No engine/model
quality filter. All starting histories and completed games are legal. These
new stage families are not natural openings or a universal holdout from every
inherited historical pretraining record; that exposure is incompletely known.

All arms48colourpaired games,16sim/max4/G0/value_scale0.1/maxvisit_init50,
CPU1FP32/eagerB1/max400plies/claim_draw=True/seed19. Internal890s and outer900s
whole-arm limits, no extensions. Primarygate score>0.60/bootstrap lower>0.50/
unknowncaps<=0.10. Outcome caps diagnostically0.5 only; this suite had zero caps.

| Arm | Wins/actual draws/loss/unknown | Score |24pair conditional bootstrap95%|Arena wall|NN evaluations|
| --- | --- | --- | --- | --- | --- |
|candidate-vs-initial|12/22/14/0|0.479167|[0.40625, 0.5520833333333334]|200.271076|92929|
|candidate-vs-stockfish512|2/3/43/0|0.072917|[0.020833333333333332, 0.13541666666666666]|52.557745|23081|
|initial-vs-stockfish512|3/6/39/0|0.125000|[0.052083333333333336, 0.20833333333333334]|56.165328|24249|

All144complete games/12998legalplies and durable
permove traces verified. Primary conservative Hoeffding95%
[0.20194535562792215, 0.7563879777054112] beside
conditional bootstrap; prior exploratory checkpoint/method choice and unusual
stress distribution limit inference. PairedSF512 candidate-minus-initial
score-0.052083333,95%
[-0.11458333333333333, 0.0]; no external improvement demonstrated.

Stockfish19/Threads1/Hash16MiB/512requested nodes; actual per-ply overshoot and
totals retained. This stronger budget differs from oldSF32, scores are not
merged; neural simulations and engine nodes are unequal compute. Scores0.0729
and0.125show substantial weakness even at this bounded reference, not a strong
engine or general Elo result. Wholewall/NNcalls/permove median/p95 and engine
work recorded; differing game lengths are not an isolated model speed ratio.

Latest512tests/zero skips43.79s at preceding source; this stage adds only frozen
book/protocol/report, numerical runtime unchanged. AppleMetal/CUDA unavailable;
real MLXCPU support retained. No paid allocation, existing4CPU/16GiB only.
[Exact evidence](UFUK-offline-control-evidence-20261004.json) preserves every
command/fullgame/trace/audit plus original checkpoint provenance. Binary
checkpoint/data remain locally retained; Release upload400means remote backup
incomplete. Same-target longer training is not justified merely by reduced CE.
