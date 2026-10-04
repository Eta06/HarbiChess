# UFUK: prospective broader policy/value/search diagnostic

Recorded after the failed overall480game teacher-strength trial, BEFORE any new
reference queries or diagnostic searches. That failure remains failed; this
diagnostic neither extends the old arena nor replaces it with an easier result.

Question: on these now-observed development roots, does the newly trained policy
rank better moves, and does deeper search or replacing inaccurate leaf values
improve choices? Stronger Stockfish reference values isolate a diagnostic upper
bound under that teacher; they are not our learned evaluator, a production
selector, terminal outcomes or evidence of self-learning.

Freeze all48 full-history roots in their existing order from bookSHA
d3b9a49e3c60cdce62a672fbfc8ae8ab2a5167ed38cc5781432a8559e0891765.
Source5d2e95c56c1a0367b18157a734622d8372cafdc1; original18f and primaryseed26
candidatee8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03.
No root filtering for quality, source substitution, alternate checkpoint,
training gradients, data generation for training, promotion or final Elo claim.
The diagnostic may include source roots with inherited teacher/pretraining
exposure; its findings are conditional, not a new heldout strength evaluation.

At each root: fresh Stockfish19 Threads1/Hash16/UCI_ShowWDL nativeSTM reference,
32768 requested nodes unrestricted and SEPARATELY for EVERY legal forced root
move, with ucinewgame/cleared state per query. Retain last coherent completed
unbounded score/WDL/PV packet plus actually spent node totals. Do not fabricate
root move expected scores from missing MultiPV entries or zero-node terminal
claims. Legal terminal continuations may be naturally resolved by the engine.

Evaluate unchanged original and candidate raw policy choices plus candidate
FullGumbel16,64,256sim with max4 roots/Gumbel0/value_scale0.1/maxvisit_init50.
Add candidate16sim with identical priors but fresh1024node Stockfish leaf WDL
Q=W-L. The latter changes only the search leaf-value evaluator and is an
expensive causal diagnostic, not a success claim for a learned chess model.
All chess history/STM alternation/claim_draw=True semantics retained.
Rotate search-arm execution order by root; alternate model raw-evaluation order.
Report full priors, true legal reference ranking/coverage, raw neural root Q
versus native32768rootQ, selected moves and teacher expected score, search wall,
neural calls, oracle calls and actual nodes. Whole CPU1 FP32, no competing
training/benchmark. Per-search wall excludes separately recorded teacher
reference collection and loading; whole parent wall includes them all.

Budget:1800WHOLEseconds including imports/models/references/search/output,
2500engine queries and80000000actual engine nodes,25000neural evaluations;
owned process-group stop if memory current>15GiB or free disk<8GiB. Reference
watchdogs use remaining deadline, at most15s/query. Limits checked between
queries/searches and parent enforces absolute wall; bounded final-unit overshoot
of count ceilings is reported and FAILS completeness, never grants new budget.
No retries, additional roots, extension or alternative gate if incomplete.
After completion, a separately recorded300WHOLEsecond independent audit checks
every root/move/reference/query, selected-score/STM/node totals and recomputes
the fixed statistics from persisted rows. No extra engine calls or gradients;
its cost is outside the1800collection/search budget and cannot extend that run.

Five prospective contrasts on selected forced-reference expected score:
candidate raw minus original raw; candidate16minusraw;64minus16;256minus16;
oracle16minuslearned16. Each uses50000 whole48source-game bootstrap resamples,
seed20261029, linear empirical quantiles, Bonferroni familywise0.05/5 (99% each),
ordinary95% descriptive and conservative Hoeffding intervals. Conditional
positive mechanism evidence requires mean>=0.02 and adjusted lower>0, complete
all48roots/queries and legal/hash integrity. For depth alternatives ALSO median
paired wall64/16<=5 or256/16<=20 respectively. Oracle-positive is diagnostic
only, irrespective of cost; its own compute cost remains fully recorded.
Small sample, saturated teacher WDL, imperfect finite-budget references and
observed-suite bias may make results inconclusive. Negative results stay
negative; no absence-of-significance claim proves two mechanisms equivalent.

Next mechanism is chosen from measured results and requires its own prospective
learning, retention, whole-speed and independent final-game protocol. Greater
teacher target quality, more search, better loss or stronger raw-policy ranking
does not establish genuine closed-loop improvement or a novel publication.
