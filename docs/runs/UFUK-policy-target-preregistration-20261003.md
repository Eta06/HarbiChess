# UFUK policy improvement targets: before reference queries

Two self-learning gates failed. The earlier96-position diagnostic used16sim/
Gumbel0 selected moves, whereas real updates used64sim/Gumbel1 soft targets.
Measure the actual policy improvement target instead of assuming these equivalent.
Read-only diagnostic; no optimizer updates or new self-play/promotion.

Frozen model18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae,
unchanged best native bootstrap. Native dataset manifestb34a3870… fixed before
collection. Freeze128 full-history positions from16 teacher-validation families,
8perfamily: first four integer-game-index jobs of each engine-engine/neural-engine
role, one uniform random labelled position pergame, alternating STMwhite/black
within each role,seed20261010. Exact2STMwhite/2black perrole/family. No score/model
filter; record actual neural-actor colours rather than assuming their balance.
These are existing development validation families, not independent test families.

Search64/max16/Gumbel1/value_scale0.1/maxvisit_init50, perposition seed
`20261010:<panel index>`. Keep all legal priors and improved `action_weights`, visits,
chosen action and child mean values. Actor self-play first moves may sample these
weights; later select action, but learner policy always receives the soft target.

For **every legal root move**, fresh-TTStockfish19/one thread/Hash16MiB/32768nodes,
same native STM coherent unbounded WDL/CP packet semantics as the previous probe;
actual nodes and full histories logged. No top-policy filtering. Restricted oracle
scores remain noisy finite-budget references, not exact minimax or actual games.

Primary: mean improvement in reference expected score of the soft search target
over the raw legal policy≥0.01 with16family bootstrap95% lower>0. Report conservative
bounds, all family effects, raw/search entropy and KL(target||raw), near-best mass,
chosen-action references and separately tanh(CP/400) auxiliary diagnostics. CP is
diagnostic only, not a changed network WDL meaning. Do not switch primary after
seeing saturated WDLs or favourable CP. Failure means this target teacher did not
meet this diagnostic; passing still requires game-level learning improvement.

1800s total wall cap,15s engine-query watchdog,one Torch and engine thread,existing
4CPU/16GiB only. May overlap registered depth learner without an isolated speed
claim. Stop on malformed/nonfinite/bounded/resource/protocol failure; preserve
every packet/partial result and report incomplete, no silent skipped positions.
Require128 completed positions, legal full histories, exact frozen panel hash,
normalized raw/target probabilities and all128×legal-count queries accounted for.
Log source/script/model/data hashes, wall/actual nodes/CPU/RSS and Git/Space evidence.
Release400 still blocks binary backup; no new paid resources or Laya integration.
