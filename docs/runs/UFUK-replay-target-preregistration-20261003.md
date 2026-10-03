# UFUK actual replay target quality: before reference queries

The128 native-validation target diagnostic passed; actual self-play96games
strength failed. Distribution shift may separate these results. Audit **stored**
raw and search probabilities in the real replay, not a new stochastic search.
No updates, external labels used for training or strength/promotion claim.

Frozen source `artifacts/ufuk-selfplay-20261003/replay`: all six immutable
checksummed version2/encoder1/action1/target12 shards,three generations/96games.
Include train and the three capped validation games in this diagnostic; unknown
outcomes stay unknown. Verify every shard's snapshot hash against its generation's
actor checkpoint and frozen48-opening book. Never reinterpret native WDL as
terminal replay outcomes. Source model snapshots0..2 remain unchanged.

Freeze96 full histories: each of48starting-opening families has two games;
choose one uniform recorded white-STM row from earlier game,one black-STM row
from later game,seed20261011. First freeze panel and store hashes before querying.
No position-quality/model/reference filter, require stored normalized raw policy.
Canonical action indices map to the actual legal UCI moves at each exact history.
These are existing training families, not an independent generalization test.

Every legal root move queried atStockfish19/Threads1/Hash16MiB/32768nodes,fresh TT,
coherent unbounded native STM WDL,15s watchdog. Record actual nodes,reference
packets and selected replay action. Primary stored-target−stored-raw expected
score mean≥0.01 and48family bootstrap95% lower>0. Conservative bound, per-position/
family effects,entropy/KL,CP/mate and actual termination support separately.
Finite-budget references remain approximate, not minimax or played games.

1800s wall cap,existing4CPU/16GiB,one engine thread/no GPU/no paid resources.
Stop on malformed/nonfinite/bounded/protocol/resource failure; preserve partial
outputs and report incomplete rather than skipping. Require96/96positions and
alllegal query counts/node totals audited. No reruns to choose favourable results.

Only a passing actual-replay target gate permits a separately preregistered
offline comparison of head-only vs extra policy representation trained on those
self-generated labels. The current shared trunk also contributes to base value;
unfreezing it cannot preserve value by freezing heads. An independent policy
residual branch is a prospective mechanism, not a proven improvement. Passing
offline losses still requires new game-level strength evidence and fresh closed-loop
self-play before claiming sustained self-learning. Preserve all failed weights.

Record important results Git/Space, exact source/model/replay/panel hashes and
resources. Release400 prevents new remote binary backup; no false completion claim.
