# UFUK: outcome value calibration with an anchor-only control

Before new data selection, updates or matches. Six-generation narrow policy
self-play failed: final score0.541667, bootstrap95%[0.46875,0.614583],9unknowncaps;
SF score0.34375 versusinitial0.364583. Preserve this failure, no promotion or
extension. Earlier oracle diagnostics did not support value as the sole bottleneck.

New hypothesis: late observed outcomes can improve continuation evaluation while
an existing native-reference anchor limits forgetting. This is uncertain, and
native stronger-policy WDL and weak-policy observed returns are different
conditional values. Mixing them is an experiment, not semantic equivalence or a
newly invented RL method. No new engine labels/actors, paid compute or Laya use.

Both arms start native18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae
with reset optimizers. Freeze shared/policy parameters, train only value-specific
modules: value_*, invariant_value_*, material_value_*, global_value_* and
plastic_value_*. Full/masked policy logits must remain bitwise identical.
Within-task full native optimizer/Torch sampling RNG/cursor/input-cache resumes
are verified in fresh processes; weights-only initialization is not full resume.

Use immutable audited192game narrow self replay, only terminal-known games and
their final32 recorded preterminal positions. Whole actor family index
game_index%48%4==0 defines12heldout families; remaining36train. Exclude any
training current position overlapping heldout current positions, keeping heldout
rows unchanged. Require>=1000train rows,>=400heldout rows,>=16heldout terminal
games and>=6heldout families; otherwise stop data qualification without training.
No capped/unknown outcome substituted or engine adjudication.

Existing native teacher training split only, same36allowed actor families;
exclude all own heldout currentpositions. Uniform10000row cap, seed20261016,
exact selection/metadata/cache hashes retained. Existing20knative validation
cache is observer-only, never training. Prior initial model saw all original
teacher training families: this additional-stage family separation is not an
independent from-scratch test or general Elo.

Control and candidate sample identical native32row and self32row mini-batches:
native uniform rows; own uniform game then uniform eligible row. Loss control
0.5*native valueCE +0*self valueCE; candidate0.5*native valueCE+0.5*self valueCE.
Both compute both forward losses; no policy gradient. This controls native
anchor updates and isolates adding observed self outcomes. Record identical
sample/TorchRNG traces at matched steps. BothCPU1/FP32/AdamW5e-5/wd1e-4/clip5,
seed20261016/max4000steps/every250eval/patience1000/1800s perarm.
Shared input preparation cap600s; record separately, conservatively charge both
arms; native observer loading included in each training timer.4CPUquota/16GiB,
no GPU or additional paid allocation. Stop nonfinite/hash/source/frozen-policy
mismatch or native valueCE>initial+0.03 or native QMAE>initial+0.02.

Select minimum heldout game-balanced observed valueCE among valid checkpoints.
Qualification: candidate improves initial by>=0.10 and anchor-only control
by>=0.05; native retention and exact frozen policy pass. All stopped/failed
arms/checkpoints remain retained. No arena if qualification fails and no lucky
extension/seed search. Neither loss decrease nor native retention alone is strength.

Before first update freeze24fresh32ply roots: parent28ply narrow-self arena book,
four uniform legal suffix moves/seed20261016, exclude native labels and all prior
complete arena current positions; no engine/model-quality filtering. Same
development families, no independent Elo or paper novelty claim.

Only if training qualification passes, candidatevsanchorcontrol and eachvsSF32,
48colour-paired games/arm,16sim/max4/Gumbel0/value_scale0.1/CPU1/eager/batch1,
max240plies/claim_draw=True/900s perarm/samebook+seed. SF19/Threads1/Hash16MiB,
32requestednodes and actual per-move node counts. Primary model score>0.60,
24pair bootstrap95%lower>0.50,caps<=0.10; report conservative bound and paired
SF delta. This tests the added own-outcome value contribution with anchor control,
not pure self-play from scratch, Stockfish/AlphaZero strength or teacher exceeding.
