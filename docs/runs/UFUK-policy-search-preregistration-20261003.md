# UFUK direct policy vs own search: before arena

The96-position diagnostic found raw regret0.00616 vs16sim0.01925, but paired
eight-family bootstrap crosses zero. Test actual games before changing defaults
or declaring that search is harmful. This compares inference selection, not
self-learning or a novel architecture. Keep previous failed training results.

Single frozen network: UFUK native-validation-selected18f2aae5… step9250,
fullSHA`18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae`.
Candidate selects max legal prior, deterministic UCI tie break, one evaluation
per move/no tree/value decision. Opponent uses identical weights FullGumbel16,
max16/Gumbel0/value_scale0.1/maxvisit_init50. Then compare each selector vsSF19
Threads1/Hash16MiB/requested32nodes, same roots/colours. No optimizer updates.

Freeze24 **20-ply** roots before games, extending previous16ply UFUK arena roots
by4 uniformly sampled sorted legal moves,seed20261008; reject terminal and
already recorded root only, no engine/network-quality filter. Exclude all native
dataset positions and all prior arena positions. Components exclude teacher
train/validation but reuse24 development families; not new independent families
or public Elo. Store parent hash, generation algorithm and final roots in Git.

Three48-game arms: raw-vssearch,raw-vsSF32,search-vsSF32. One Torch thread,
seed20261008/max240totalplies/900s arm limit, existing4CPU/16GiB; no paid resources.
Record full legal histories/source hashes/model hashes/NN counts/wall, preserve
unknown caps explicitly. Actual SF nodes remain unrecorded by existing arena;
sim/node and raw-vssearch compute unequal, no equal-compute claims.

Primary direct-selector strength gate: score>0.60,family bootstrap lower>0.50,
caps≤10%. Also report conservative bounds, SF paired delta and separately both
SF walls and NN evaluations. Any strength gate failure keeps search default;
do not select successful roots/change thresholds/repeat until lucky. A positive
result would be bounded development evidence for an inference choice, not proof
of Stockfish/AlphaZero strength or autonomous learning. Stop on engine/protocol/
nonfinite/resource/integrity failure and preserve incomplete outputs.

New `--candidate-policy-only` is diagnostic only/default off; searched opponent
unchanged. Test real legal moves, one backend call per raw candidate move and
zero tree calls; existing history/colour/cap tests must still pass. Run at a clean
fixed source checkout, no shared-tree edits during games. Log final result/commands
and hashes Git+Space; binary model backup remains blocked by Release400.
