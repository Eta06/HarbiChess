# UFUK: tested data controls; broad collection still running

Infrastructure result only, recorded before broad-data production training.
Main strong/fast/reliable self-learning and Stockfish/AlphaZero-level goal remains
unmet. No new learning, completed broad corpus, arena result or promotion claim.
All earlier failed experiments and v3 incomplete collection remain preserved.

Immutable07:11:24UTC progress snapshot:14,455/20,480completedv4games,
414,423labels,13,513,518,566actualStockfishnodes, emptyquery stderr. Source8fa4f29,
same5120sourcePGNroots/seed20261025, opposite neuralcolours at every root. The
400,000label threshold alone does not complete the planned collection. The v3
start/failure/fix/idle wall remains charged to absolute08:20:19UTC deadline;
no new budget, queries, data mutation or training extension is introduced.

Found and fixed a CLI wiring error: oracle_train accepted --prepared-cache but
did not forward it into run(). Actual production packed throughput experiment
used direct run(prepared_cache=...) calls and remains valid. The new subprocess
CLI test performs an optimizer update and confirms prepared manifest SHA plus
all12npyinputs in the full native checkpoint. Optional cache/raw configurations
remain deliberately incompatible for full resume. Historical source pins and
original80511arrays/raw bytes/checkpoints are unchanged.

New controlled_oracle_train.py provides two data arms with ONE common immutable
validation source. Every training-key in either arm must be present in that
common source's training union; validation removes all those keys before any
cap. Named family ranges are nonempty/disjoint/fully covering, sampled by fixed
seed pergroup, and the exact indices hash is recorded. For merged schema1,
anchor families<100000 and broader validation200000..299999 are separately
reported. Identical panel indices/tensors for both arms are tested, including an
adversarial overlap fixture and refusal when a training key lacks union coverage.

Both soft policy and nativeSTM WDL heads and used shared trunk train jointly.
The21historically unused material auxiliary parameters remain frozen and hashed.
Equal group mean(policyCE+valueCE), minimum improvement0.0001, selects the
checkpoint/patience anchor. This is an explicit training objective, not a
chess-strength or retention gate. Per-group policyCE/valueCE/QMAE/top16coverage
remain visible. Retention, replicate and strength acceptance will be separately
preregistered after actual no-gradient baselines, before any optimizer update.

Native checkpoints preserve model, AdamW, RNG, sampled-index hash chain, cursor,
selection/evaluation state, runtime/config/code and ALL source/raw/array hashes.
An actual fourupdate test compares uninterrupted training against two updates,
then fresh Python process resume: weights/optimizer/RNG/sample trace/validation/
selection bitwise identical. CLI and changed-seed profile refusal tested. Frozen
auxiliaries exact; used parameters actually change. This is full within-Torch
resume, not weights-only transfer or cross-framework optimizer equivalence.

Root selection now optionally excludes entire source-game IDs and supports an
explicit arena split. The default train/validation book fields and selection
remain unchanged when omitted. New seed/root from an already used source game
cannot become an independent evaluation family. Tests verify default contract,
actual legal full histories, source exclusion, insufficient candidates refusal
and invalid named counts. No new arena book has yet been frozen.

Tests actually run: prepared CLI/native suite5passed5.84s; new common validation
suite7passed9.66s; root-source isolation suite7passed0.56s. Ruff passed each
changed group. First common-validation test run1failed/6passed because its
fixture tried overwriting an immutable published file; corrected by constructing
an explicitly separate altered arm, no production integrity check weakened.
Original failure and style correction logs retained. Last complete suite before
these additions559passed/47.35s/zero skips. The new complete suite is pending;
there is no fabricated cumulative whole-suite pass count. Real MLX LinuxCPU
gradients/updates and portability were already tested; AppleMetal/CUDA unavailable.

Pinnedf4c10e5postcollection controller waits for completed20480game coverage,
then runs independent everygame/everyrow audit3600wholewall, full tests300whole,
version1merge3600whole, every104input-exact preparation3600whole and cold
baseline600whole. All phases have resource guards15GiBmemorycurrent/8GiBfree,
owned-process-group termination, durable command/stdout/stderr/result receipts,
no overwrite and no budget reset. The controller's status at this snapshot is
WAITING for collection; these prospective phases are not completed work.
Trainingcontrol old80511+its existing packed cache, candidate versioned merged
source; original80511copy bytes remain exact and all heldout keys protected.
The cold baseline will use seed20261026/20,000rows pergroup, equal positions
for both arms, target entropy and policy/value/Q metrics, ZERO updates. The
fixed400,000rawrow collection requirement is not a unique/postfilter row claim.

[Independent strength/speed protocol](UFUK-broad-history-strength-preregistration-20261004.md)
committed9cd0d25before root selection and new production updates:48newsource
games/96colourpairedgames perarm, five comparisons including Stockfish512,
identical16sim/max4search, fixed wholewall/node/resource budgets, adjusted
pair-bootstrap uncertainty, explicit cap/worst-case/speed/strength requirements.
It cannot turn a weak Stockfish512 reference into general Stockfish-level power
or teacher pretraining into genuine self-learning. No paid resources purchased.

One meaningful file/commit, prefixUFUK, author AND committer Emir Tunahan Alim
<emrtnhalim@gmail.com>, no co-author. GitHub push actorEta06 is separate. Four
training source/test files actually downloaded from remotef4c10e5 and matched;
root source/test/strength preregistration downloaded from remote9cd0d25/matched.
Space updatedsequence24: new status exact, all110prior nontarget blocks unchanged,
previous status heading retained. Historical151child documents untouched.

Provider HuggingFace paper search returned UNAVAILABLE for two new queries;
no findings fabricated. Previously retrieved primary AlphaZero/Mctx/Lc0/KataGo/
ChessBench mechanisms remain cited in PORT/MERCEK research. Concrete current
decision is broader true-history coverage plus shared old/new controls, not
mandatory Laya integration, an unmeasured architecture change or novelty claim.

New remote binary backup remains INCOMPLETE: authorized Releases uploads fail
400BadContentLength through current network path. Five verified local archives
preserve earlier binaries; broad input/v3/v4/postcollection artifacts are not
yet in a sixth completed supplement. Exact UTF8 scientific evidence in Git and
remote text hash restoration do not back up binary models. Local files/source
pins and old failed records retained; no credentials/proxy/storage bypass.

Source mapping:46ec80c oracle_train.py;d065f0d test_prepared_oracle.py;
3896e25 controlled_oracle_train.py;f4c10e5 test_controlled_oracle_train.py;
46f36cd history_openings.py;cb0567c test_history_openings.py;
9cd0d25 strength preregistration. Full hashes and exact fixed UTF8 commands/logs/
source proofs/controller profile/progress snapshot/Space readback are in the
paired evidence JSON. Growing live collection/progress logs are excluded from
this immutable midpoint evidence and will be recorded at their terminal boundary.
