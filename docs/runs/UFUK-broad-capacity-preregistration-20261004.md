# UFUK: broader-corpus joint capacity and optimisation, prospective protocol

BEFORE new transfer, updates or final-game roots. Earlier narrow-corpus width/
context/self-learning and current broad Stockfish/depth-diagnostic failures
remain failed. Broader two-seed training improved heldout loss and direct
model-vs-model strength, but SF512score remains0.072917. More search alone lacked
conditional quality evidence. Capacity is a new hypothesis, not a proven cause.

Start BOTH arms from primary broad candidate e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03.
Control72,497parameter native architecture versus function-preserving widened
shared trunk16->64 and independent value tower16->32, conventional active-feature
width-v1 transfer (existing tested source), same pairwise head/rules/104history/
4672action/STMsoftWDL. All used trunk/policy/value parameters train; unused
21material readout parameters remain frozen. No claim of novel architecture.

Width transfer is schema1 WEIGHTS-ONLY, optimizer reset; new control also starts
with reset AdamW. Neither is resumed old max30000training. Thereafter every arm
retains full native model/Adam/RNG/source+input hashes/cursor/index trace. Require
actual pause1000/fresh-process full resume, and isolated1000->2000 replay matches
real production model/Adam/RNG/sample-chain/cursor bitwise for each of four runs.
Replaying audit updates are separately charged and excluded from learning.

Immutable raw merged667369rows, train530036, exact prepared104history input
manifest8c9a06de8be2c1f15a44569cd7bfd0fbc234907242f2145dfe8f3dee5a3fb4f5;
no train cap. Common old/new heldout20,000rows PERgroup, same exact rows across
arms withinseed, all training-union key overlaps removed first. Both use same
completed teacher corpus, not independent-data replication. Seeds20261030 and
20261102; each seed has its own transfer RNG and matched row sampling streams.
Order narrow30,wide30,wide1102,narrow1102; CPU1FP32deterministic/AdamW1e-4/
wd1e-4/clip5/B64/both head weights1/max40000/eval2000/patience20000.
7200WHOLEseconds EACH including imports/hash/data/startupeval/pause/resume/
checkpoints. Same maxima/stopping rule, not equal actual work if early stopped.
Select minimum equal-group mean policyCE+valueCE (existing min improvement1e-4).
No post-game selection, learning-rate tweak, alternate seed or extra updates.

Preflight whole300s: both seed transfers caller RNG preserved;18existing fixed
true-history full4672policy/WDL/legal-mask initial function and real MLXCPU parity,
atol/rtol2e-5 unchanged; olde8 unchanged. After success, separately whole120s
disposable six-step gradient probes per seed/arm verify new-width features/readout
learn and frozen21stay bitwise;24audit-only updates/1536presentations excluded.
Separately whole300s fixed inference+full16sim root speed check, rotatedarmorder:
18full-history probes/10warmup/200maskedB1rounds and four16sim searches/root/arm,
include encoding/rules in search timing. Median wide/narrow <=2 for BOTH metrics
before production; setup excluded/included fields explicit. No concurrent
training/benchmark; AppleMetal/CUDA/BF16device unavailable, no skips conceal this.

Both seeds separately require selectedwide gains versus their OWN initial:
new policyCE>=0.10, new valueCE>=0.06, new QMAE>=0.03; anchor policy<=+0.03,
value<=+0.02,QMAE<=+0.015; selected macroCE improvement vs selected same-seed
narrow>=0.05. Need normal max/patience stop under whole budget, finite/full
native integrity, fresh resume audits and trained selected Torch/MLXCPU parity.
No average substituting for a failed seed. Library training source pinned
f4c10e549c82fa35e7d595ccada90a15dccfd613 (unchanged tested common-panel trainer).
Production resume audit whole600/run, complete-state audit whole1200; separately
charged. Source currentfullsuite587PASS/zero skips retained; targeted tests only
if actual production implementation changes. All checkpoint binaries preserved.

BEFORE updates freeze48NEWdistinct source-game roots, seed20261031, same bounded
32MiBCC0PGN prefix/standard/fullhistory/bothratings>=2000/baseclock>=60/root
plies16,24,32,48,64,96/atleast8recordedpliesremaining. Exclude ALL5120teacher
sourceIDs plus48observed prior strength sourceIDs, ALL old/new teacher position
keys, prior exclusions and every observed480game and diagnostic query position
key. No engine/model/quality filter, new download, source substitution or lower
criteria. Freeze600whole/independent original-PGN audit300whole; no query/update.

If BOTH learning seeds and integrity pass, use ONLY PRIMARYseed20261030selected
wide/narrow; five96game arms: wide-vs-e8,wide-vs-narrow,wide-vs-SF512,narrow-vs-
SF512,e8-vs-SF512. Same48newfullhistory roots/two colours/CPU1FP32/eager/max4/
Gumbel0/16sim/SF19Threads1Hash16/512requested+actualnodes/max400TOTALplies/
declaredclaim_draw=True. Each3600WHOLEseconds; incomplete96fails. Unknowncaps
diagnostic0.5 plus adversarialbounds, never terminaldraws or outcome labels.
Three primary48family50000bootstrap/seed20261031/linearquantiles/Bonferroni
98.333% intervals: BOTH directscores>0.60/adjustedlower>0.50/worstcapbound>0.50;
wideSFscore-minus-narrow>0.10/adjustedlower>0, and ADDITIONALwideSFscore>=0.25;
caps<=5% everyarm. Independent every-move/history/actualnodes audit300whole.
Final fixed inference/search speed same preflightmethod300whole;wide/e8 and
wide/narrow median<=2 BOTHmaskedB1 and full16sim root, separate game-throughput
counts/wholewall. Unequal architectures/sims/nodes are not equal actual compute.

Stop owned groups if memory current>15GiB/free disk<8GiB; no new paid resources,
deadline extensions, retries to replace failure, promotion or deletion of old
data. Even complete learning/strength/speed success is a bounded teacher-stage
gain, NOT yet Stockfish/AlphaZero level, general Elo, teacher superiority,
genuine closed-loop self-learning or paper originality. Next self-learning
mechanism needs prospective actual initial/final game and persistent replay/
optimizer gates; raw imitation does not automatically outperform its generator.
