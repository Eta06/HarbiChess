# UFUK: broader complete-history teacher pretraining

Prospective protocol before root selection, teacher collection or new updates.
Main strong/fast/reliable self-learning and Stockfish/AlphaZero-level goal remains
unmet. Joint-context, sparse, search and older self-learning failures stay failed.
Packed native infrastructure passed separately; it is not a chess-strength gain.

Hypothesis: previous narrow64family teacher coverage restricts useful learning;
many distinct real-game middle/endgame starts may improve out-of-family policy
and value together at fixed model/search. This is controlled teacher pretraining,
NOT self-learning or proof that imitation exceeds the teacher. Ordinary PGN
parsing/teacher distillation/mmap do not establish publishable novelty.

Primary source https://database.lichess.org/ verified CC0 game exports. Bounded
32MiB prefix of standard rated September2026 PGN archive, HTTP206; sourceSHA
d629023c6ceb487afcd676cc840e744a1f97b8f8782e9f0af3f846998fa205f6,
full compressed archive29224520887bytes was NOT downloaded. Existing zstd1.5.7
decompressed partial prefix to229MiB; explicit premature-end exit1 retained.
Final incomplete PGN record rejected, never repaired/fabricated. Header-only
research found102354numeric-rating/time records,15868with bothratings>=2000
and baseclock>=60seconds. This is a chronological archive prefix, not a uniform
population sample. No model score/teacher score/result quality selection.

Freeze5120unique source games/roots, seed20261025. Eligible standard initial
state/no custom FEN/variant, legal complete mainline, terminated PGN result token,
bothratings>=2000/baseclock>=60. Choose ONE full prefix per source game from
16/24/32/48/64/96plies with at least8remaining recorded plies, uniformly using
seed and game identifier. Reject drawn/terminal/invalid roots, duplicate game ID/
four-field root key and roots present in native80511source/current prior arena
roots. Keep every UCI move from true standard initial state, actual clocks of
chess halfmove/fullmove/repetition/EP reconstructed; player wall clocks are not
model inputs. Do not use incomplete evaluation FENs or reset unknown history.
Shuffle complete eligible source games once, train4096/validation1024disjoint
game families. Freeze root/source/book/exclusion hashes before teacher labels.
Shared common opening prefixes are expected; no complete inherited-pretraining
holdout claim. If5120valid distinct games unavailable, collection is INCOMPLETE,
do not lower rating/clock/coverage criteria or enlarge download after results.

Each family4trajectories:2engine/engine plus2original18f/neural-vs-engine with
balanced neural colours, all plies labelled. Existing SF19binarySHA
0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19,
Threads1/Hash16/MultiPVmin(4,legal)/32768requestednodes perlabel, coherent exact
completed MultiPV/STM WDL/cp policytemperature100. Identical original actor
stochasticity; actor weakness cannot turn observed outcomes into teacher labels.
Stop after32new plies or real claimable outcome. Up to655360new rows,
not assumed actual count. Original80511raw/replay/models unchanged. Extend
collection with versioned optional continuation bound; legacy default160plies,
metadata/jobs/resume contract unchanged when option omitted. Strict invalid
bound/terminalroot refusal. Code/weights/engine/book metadata and full histories
persist; completed pergame native oracle schema1files remain immutable.

Existing4CPUquota/16GiB/32Gdisk/noGPU, no paid resources. Collection10800WHOLE
wall seconds for one fixed stage, including import/pool/model/engine setup and
failures; max4spawn workers/1CPU each. Completed games are resume boundaries;
partial queries/failures retained but never training rows. Parent records charged
wall; interruptions use REMAINING registered budget, no fresh10800reset. Stop
early for invalid reference/history/timeout/low disk(<8GiBfree)/resource violation.
No automatic retry with more nodes/new seeds or relaxed packet semantics.
After collection audit every source game/hash/row, actual nodes/wall/colour,
family leakage, outcomes and unknown caps. Collection succeeds only completed
planned20480trajectories, alllegal native targets, >=400000new labelled rows,
no source/hash corruption. Otherwise preserve INCOMPLETE/FAILED and stop this
protocol before training; no lucky budget extension.

Build a versioned merged raw-source manifest from unchanged old80511 and new
native teacher files. Preserve original bytes/parent provenance. Namespace family
IDs, exclude new training rows overlapping ALL old heldout position keys and new
heldout keys; decoder also removes exact training/validation position overlap.
Never relabel an old validation row as training. Record dropped counts and
post-filter per-family coverage. Prepare optional mmap once with EVERY full104
tensor checked against BoardEncoder. Preparation3600whole-wall seconds/1CPU,
no overwrite, source/array hashes, stopped incomplete cache cannot load.

Training controls, selection and independent strength protocol will be separately
frozen BEFORE ANY update after exact actual collection audit, dataset hashes and
baseline no-gradient losses are available. They must compare original18f fixed
architecture at equal update/wall/search budgets, complete native optimizer/RNG/
input/sampler resume, untouched original holdout and new whole-game holdout,
policy and WDL/Q rather than policy-only loss, fresh colour/openingmatched games
and Stockfish controlled resources with confidence intervals. No model promotion
or self-learning claim from improved teacher CE. Fixing a training protocol now
around an assumed nonexistent row count would conceal collection uncertainty.

Important boundaries logged in Git and Space; source/target/command/runtime
receipts retained. New binary Releases upload remains400BadContentLength blocked;
local verified archive is not remote backup. No new storage destination or
credential/proxy bypass. Decision/LLM dataset ideas remain later research; full
legal/action/search/WDL provenance can support them, but no teacher-surpassing
or LLM chess-strength outcome is assumed.
