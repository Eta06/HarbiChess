# UFUK: independent broad-history strength and speed controls

Prospective protocol before selecting the arena roots, before any broad-data
optimizer update, and before playing these games. The collected teacher labels
are not self-learning; the main Stockfish/AlphaZero-level goal remains unmet.
Training/selection controls will be separately registered after collection audit
and the no-gradient old/new baseline. This document fixes the subsequent games
and prevents choosing an easier suite or budget after seeing a model result.

Hypothesis: broad teacher pretraining, if it first passes its registered learning
and retention gates, improves actual play over the original18f model and the
matched native-data training control, at identical architecture and search.
Only the first registered primary training seed supplies arena checkpoints;
there is no best-of-seeds selection using game results. A second training seed
must satisfy its own registered learning gates, not replace a failed first seed.

After completed20480-game collection passes every-row/source/history/colour/
reference audit, freeze48 distinct source-game roots from the already retained
CC0 September2026 chronological32MiB PGN prefix. Use selection seed20261027 and
the existing bothratings>=2000/baseclock>=60/standard/legal/complete-record/
atleast8remaining-recordedplies filters and root lengths16/24/32/48/64/96.
Exclude ALL5120source-game IDs in the broad train/validation book, ALL position
keys in old80511 and completed broad source rows, and every prior recorded arena
position key. One root per distinct new source game; no model, engine score,
result quality or selected-checkpoint filter. The source record SHA, full true
UCI history, six-field FEN, source/exclusion/book hashes remain immutable.
No resetting FEN history or inventing clocks. Selection600wholewall seconds,
including imports/source checking; independent root audit300wholewall seconds.
If48 valid new source games cannot be frozen, this arena stage is INCOMPLETE;
no lower rating/clock rule, new download, seed or favourable replacement suite.
Sharing common opening prefixes is possible; this is a conditional development
suite, not a population sample, exhaustive inherited-pretraining holdout or Elo.

If both training seeds pass the separately frozen learning/retention gates,
play five sequential arms with the primary-seed selected checkpoints:

1. Broad-data candidate versus original18f.
2. Broad-data candidate versus matched native-data control.
3. Broad-data candidate versus Stockfish19 at512requestednodes/move.
4. Native-data control versus the same Stockfish19 budget.
5. Original18f versus the same Stockfish19 budget.

Every arm uses the same48 roots, both neural colours per root,96plannedgames,
Full Gumbel16simulations/max4rootactions/Gumbel0/no self-play noise, FP32 CPU
Threads1. Stockfish19 binarySHA
0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19,
Threads1/Hash16, full-strength options, no Elo/Skill handicap. Record real actual
nodes because requested512 is not exact actual work. Engine new-game boundary
per source root/colour. Uniform suite order, seed20261027, true-history moves.
Max400totalplies including the initial prefix, claim_draw=True declared engine
convention, legal moves and exact outcome/rule termination independently audited.
Unknown caps remain unknown, never terminal training draws. Existing diagnostic
score0.5 for caps is supplemented with explicit worst/best-case score bounds.
Each arm ceiling3600WHOLEwall seconds including imports, models, UCI setup,
encoding/search/rules, games, durable progress and result publication. No other
benchmark/training concurrently. Memory current>15GiB or diskfree<8GiB stops
the owned process group and retains partial traces. No new paid compute.
Timeout/incomplete96games fails this protocol; no extensions or added games.

Predefined three primary comparisons: candidate score versus original18f,
candidate score versus native-data control, and candidate-minus-control score
against Stockfish512 matched by source family and colour. Resample48 whole
colour pairs, never plies or individual colours, with50000bootstrap replicates,
fixed seed20261027. Bonferroni familywise95% intervals use per-comparison
98.333333% percentile intervals (alpha0.05/3); report ordinary95% descriptively
and conservative Hoeffding pair intervals as additional uncertainty. Dependence
and finite-suite selection limit these intervals; no general Elo claim.

Strength stage passes only if BOTH direct candidate scores>0.60, BOTH adjusted
lower bounds>0.50, and paired candidate-minus-control Stockfish512 mean>0.10
with adjusted lower bound>0.0. Caps must be<=5% in EVERY arm. Require the direct
candidate worst-case cap-score bound>0.50 too. A duplicate selected control and
initial weights is explicitly recorded; it does not become two independent
baseline discoveries. Baseline scores/nodes/wall and all failures remain visible.
The original18f Stockfish arm is descriptive; it cannot rescue failed primary
comparisons. Lower teacher CE or high CPU use never overrides a failed game gate.

Measure selected candidate/control/original masked batch1 policy+value latency
on the fixed18actual full-history probes in
docs/research/UFUK-width-latency-probes-20261004.json, SHA
1089fd0cca308c24bb040a840352d0ad18aa85a7bdcd17608133456bf08bed9f.
Threads1/FP32,10warmups/200timedrounds perprobe, alternating arm order;
encoding/setup excluded and explicitly reported. Candidate median must be<=1.10
times original and<=1.10 times control. Ceiling300WHOLEwall seconds for this
same-architecture latency experiment. Actual arena wall/plies/NN calls/node
throughput and per-move time distributions are separately reported; warmed
forward latency alone is not whole-game speed or strength. No GPU exists here.

If any learning, integrity, strength or speed gate fails, preserve FAILED or
INCOMPLETE, no promotion, no automatic extra updates/arenas/reseed. Subsequent
ideas need a new prospective controlled protocol. If all pass, this is a bounded
teacher-pretraining strength improvement on a weak Stockfish512 reference, not
Stockfish-level play, AlphaZero equivalence, self-learning, teacher superiority
or paper-worthy originality. Genuine subsequent closed-loop self-play progress
requires its own initial/final independent strength control and persistent full
optimizer/RNG/replay resume; no learned-success claim is made in advance.

Important boundaries go to Git and Space with commands/source/data/model hashes,
whole wall/CPU/memory, failed arms and exact native checkpoint evidence. Previous
binary artifacts remain preserved; the current Releases400BadContentLength
upload blockage and incomplete new remote binary backup are not disguised.
