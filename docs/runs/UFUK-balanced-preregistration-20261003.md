# UFUK balanced teacher curriculum: before collection and training

AYNA self-learning failed its fresh strength gate: final/bootstrapped5/31/4,
score0.5125,20families, bootstrap[0.4375,0.5875], no caps. Do not extend that run
until lucky. Native reference value MAE worsened0.3322 ->0.3882 while actual
held-out self-game WDL CE improved; execution and strength are separate.

Measured sampling confound: AYNA3601 reference rows contain2756 engine/engine
rows, all white STM, and845 neural/engine rows, all black STM. Actor source and
absolute colour are perfectly confounded, despite correct STM encoding. The
white-colour metadata can act as a source/quality shortcut. No claim yet that
this alone caused the failure; isolate learning on a corrected, broader dataset.

## Data, resources and fixed controls

- Reuse the48train/16validation transposition components, source CC0 opening
  split SHA019c5b4a49ac287ca03effb65b33ed97daa2e702ddd4a6972491969e27912b24.
  The validation families are development data already used for earlier decisions.
- Seed20261005,16trajectories/family:1024games total, max160plies. In each block
  of4games, two engine/engine then two neural/engine; neural colour alternates
  by family+block. **Label every ply**, both STM colours in both sources.
- Frozen neural raw-policy actor: qualified pairwise bootstrap SHA
  24679cbeacc2ad0a65e6e80ada5867bc833d60618a799270228ccf5c6800ade3.
  Teacher actors use first PV, except20% soft-policy exploration before ply40.
- Same Stockfish19/one thread/Hash16MiB/MultiPV min(4,legal),32768requested
  nodes, native STM WDL, coherent completed unbounded PV cycle, CPtemperature100.
  Every query has actual nodes/depth/full legal history. Reference probabilities
  remain separate from observed terminal replay. Unknown capped endings stay null.
- Three spawned workers/one Torch thread each, existing4CPU/16GiB only; max3600s.
  A separately recorded one-CPU arena may finish during early collection. No
  throughput comparison is inferred from concurrent runs. No purchased compute.
- Preserve immutable per-game files and checksums; stop on malformed targets,
  protocol/integrity/nonfinite/resource failure. Dataset completion is not strength.
- Learning panels capped at80000train/20000validation rows using fixed seeded
  row sampling, retained whole-family split. Audit role/colour counts, any current-
  position train/val overlap and arena-root overlap. Exclude train/val overlap as
  before; stop if a frozen arena root appears in training. Keep raw data intact.

## Controlled training arms

Both start from the same frozen bootstrap weights, explicit optimizer reset, no
architecture/encoder/action change. Batch64, one Torch CPU thread, seed20261005,
AdamWwd1e-4/clip5, max10000steps or1800s, eval every250, stop1500updates without
validation total-CE improvement>=1e-4. Checkpoint selection only on development
validation; record actual sample and unique-data counts.

- Head control: only `pair_` parameters, LR2e-4. Isolates broader balanced data
  from new trunk/value representation learning. Frozen WDL must stay identical.
- End-to-end: all72497parameters, LR1e-4. Learns policy trunk and value too.

Qualification: head policy CE improves>=0.10 vs frozen control, unchanged value
CE; end-to-end both policy and soft-WDL CE improve>=0.10. Record QMAE/top16 too.
Failure remains failed; neither arm is selected using arena scores. After a
qualified end-to-end arm, true self-play requires a **new** registration; teacher
imitation alone is not self-learning or automatically stronger than its teacher.

## Fresh root strength and speed test

Frozen24families/48colour-paired games,12-ply root continuations, seed20261005,
split SHAaec0ffe243011f3d4299b798b1608008b231e738fd35292a880e31b31086904d.
Families are component-disjoint from all teacher train/validation and AYNA's20
self-play arena families. Some components overlap AYNA's earlier32-family arena,
but these12-ply root positions never occurred in its192games. This is fresh-root
development evidence, not a wholly independent universal Elo estimate.
For short CC0 lines, explicitly marked seeded uniform legal suffixes extend to12;
no engine/candidate-quality filtering. Same roots/colours for every arm; starting
advantages may vary and both colours are paired. Report this stress-test scope.

Neural16simulation/FullGumbel0/one thread, max240plies. Each qualified arm versus
frozen bootstrap and Stockfish32; frozen control versus Stockfish32. All48legal
games/arm, caps separate. Strength improvement requires score>0.60 vs control,
paired-family bootstrap lower>0.50, caps<=10%; report conservative bounds and
all Stockfish losses. Never claim high strength from beating weak32-node engine.
Sim/node costs differ; an equal-wall test is a later explicit control.

Archive source/hash/commands/tests/compute and every failure, Git/Space links.
Release upload is still blocked; no remote model/data backup success claim.
