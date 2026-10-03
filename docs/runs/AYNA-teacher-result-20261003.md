# AYNA teacher/value diagnosis — 2026-10-03

## Result

Completed v2, 96 frozen development positions, 2,200 engine queries and
17,255,494 actual Stockfish nodes. The conditional value intervention supports
prioritizing value learning: same neural priors/search, root and leaf value
replaced with budgeted native Stockfish STM WDL, restricted move regret reduction
0.08121, family bootstrap 95% [0.03375, 0.13067]. Conservative Hoeffding interval
[-0.87911, 1] is very wide. Eight reused development families, differing oracle
compute, restricted candidate set and finite-depth reference prevent a general
strength/Elo claim. No production oracle, promotion or teacher-optimality claim.

| Arm | Mean restricted expected-score regret |
|---|---:|
| Initial raw policy | 0.100953 |
| Failed PORT latest raw policy | 0.098464 |
| Initial neural search, 16 simulations | 0.089078 |
| Initial neural search, 128 simulations | 0.043661 |
| Initial priors + engine value, 16 simulations | 0.007865 |

Teacher best move occurs in initial policy top 16 on 53/96 positions (55.21%).
Reference Q = P(win)-P(loss), root STM: initial MAE 0.906972, PORT latest
0.915691. More search reduced fixed-position regret by 0.04542 with conditional
bootstrap [0.00760, 0.08860], below preregistered 0.05 mechanism gate. MERCEK's
whole-game score gain remains unsupported; these are different measurements.
4096/32768-node best-move identity agreement is 62/96 (64.58%); differing moves
can be comparably good, so identity disagreement is not automatically bad WDL.

All 96 source games have known observed outcomes. Initial/latest observed WDL
CE 1.064766/1.269748, Brier 0.644494/0.816294. These are outcomes of weak source
players, separately reported from engine-reference labels, not optimal play.

## Provenance, protocol failure and verification

[Original registration](AYNA-teacher-preregistration-20261003.md) and
[v2 protocol repair](AYNA-teacher-preregistration-v2-20261003.md) preceded their
runs. V1 source 762d19a03d0de57e9369a4730b6f05676db7838a stopped incomplete after
1/96 in 1.570 seconds due to a bounded score. It is retained, with no conclusion
from its one row. Original failed UCI packets were not captured. Regression
demonstrates python-chess merged info retains stale bound flags; this is not proof
that the uncaptured particular packet had that cause. V2 streams the last coherent
unbounded score/WDL/PV packet, records completed iteration versus actual nodes,
and guards an unresponsive engine. No bounded value is relabelled as exact.

V2 source f1290dfeca5aabad6029937772b314a0b16bbcb9. Exact command is in v2
registration. Frozen panel SHA e2ae34e6773392d5ff8b143b2b8b081ad0eeaf0be87b65e034ae7b9e48545c23.
Initial weights a155be94c96227815a2cac6f235c06622a5fb1df93a215b4ba391be500aaaa61;
latest e6920ef586260a5cca279413d952d839ce9a9ef62d9865c6d4c5af0e4b1e7f83;
Stockfish 19 0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19.
SF one thread/16 MiB, Torch CPU one thread, full legal history/encoder/action
versions unchanged. Wall 62.2365 s, self CPU 38.9876 s, self peak RSS 263080 KiB;
engine subprocess CPU is not included in that self figure.

Protocol, perspective, full-history and search tests: 8 passed, zero skipped.
Apple Metal/CUDA device validation remains unavailable on this four-CPU host.
Raw queries/panel/positions/metadata/results remain under
`artifacts/ayna-teacher-20261003[-v2]`. Small panel/position/result evidence is
also embedded with exact UTF-8 bytes/checksums in the Git evidence bundle.
The 1.51 MB complete query log is preserved locally for Release archival.
Release asset uploads still return HTTP 400 BadContent-Length, including a
302-byte ASCII manifest; remote model/replay backup is **not complete**.

## Learning follow-up observed so far

[Bootstrap registration](AYNA-bootstrap-preregistration-20261003.md) and
[v2](AYNA-bootstrap-preregistration-v2-20261003.md) describe new transposition-
separated CC0 opening families. V1 collection failed on mixed same-depth
MultiPV rankings; completed files were not used. V2 requires one consecutive
unbounded PV 1..K cycle; failure packets are saved, original failure preserved.

V2 collection source 981605797b40b31e28d651e709482882698b5f16: 128 games,
3601 rows, 68.6804 s, child CPU 264.2124 s with four spawn workers. The dataset
manifest SHA is 33640698d2462894376b48d7afbed76ddd6b3a20544bd543985ed395690c3cd4.
48 train families/2605 rows and 16 validation families/996 rows, zero current-
position overlap. Family numbers are local to each split; opening components
are disjoint. Arena's 32 families have not been inspected by a model.

Unchanged MIHVER all-parameter learner source bfc83e9f4bc5260c0a157d6a9fbb4a4e3f05bad4
stopped at 1200 updates by the registered early-stop rule. Best step 200:
validation policy CE 3.272551 -> 3.301610 (worse), value CE 0.876855 -> 0.647972,
value MAE 0.575775 -> 0.332212, top-16 coverage 0.597390 -> 0.710843.
At step 1200 policy/value CE 5.176585/1.325180 despite low training losses.
**Qualification failed**: both heads must improve; no arena or self-learning
claim. Full process exit at 200 and fresh-process optimizer/RNG/data resume
worked. Wall 14.6748 + 47.4922 s; synthetic interrupted/uninterrupted training
has identical weights, optimizer moments and sampling RNG. Checkpoints retained.

Next hypothesis is shared origin/destination policy scoring with fewer parameters,
preserving the useful value checkpoint. It requires its own preregistration;
neither additional updates nor a threshold change rescues this failed arm.
