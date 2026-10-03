# UFUK policy self-learning: primary failed

The registered policy-only own-search phase completed with genuine optimizer
updates and full checkpoint resume, but **failed its strength gate**. Final vs
fixed bootstrap5/34/9,score0.458333;24 opening-pair bootstrap95%[0.385417,0.531250].
Required score>0.60 and lower>0.50 were not met. No promotion, threshold change or
extension. The interval includes0.5; this does not prove regression either.
Retain UFUK native bootstrap as the best measured development reference.

## Fixed design, sources and actual learning

[Preregistration](UFUK-selfplay-preregistration-20261003.md), source`0b02ee1`.
Warm start from step9250 SHA
`18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae`;
optimizer reset, not a previous-task full resume. Only16,914 `pair_` parameters
trained,55,583 trunk/value parameters frozen. Three generations×32 games,
FullGumbel64/max16/Gumbel1,seed20261006,one-thread CPU learner/four spawned
one-thread actors. Batch64/256 updates per generation/LR5e-5/wd1e-4/clip5,
rolling3/game-balanced replay. No Stockfish actor, query or label in updates.
Games start from48 fixed8-ply teacher-train histories, cycling game-index mod48;
not initial-position forced-random opening play. Existing exploration thereafter.

Generation1 source`1a1ed63c259fb40c6c8dc40ae5f75158fda87944`, process ended
normally. Separate fresh-process full-checkpoint resume1→3 source
`18819f9b929bc80c3e898575f37b503246057ee1`; the intervening change adds only the
number of supported held-out terminal rows, not model/loss/optimizer math.
Native optimizer/RNG/replay/cursor/trainable-parameter/input-checksum states retained.
Four immutable generation0..3 checkpoints preserved; final model SHA
`d5f5bbc5e43bf28ee06d558ee8f78242294ce0f9d1f17995835a52a159cf0de5`.

| Generation | Games/positions | Terminal games/rows | NN evaluations | Self-play s | Training s |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | 32/5134 | 20/2350 | 330005 | 263.939186 | 6.328789 |
| 2 | 32/4882 | 22/2562 | 314614 | 252.857854 | 6.353465 |
| 3 | 32/4869 | 25/3245 | 310063 | 239.071797 | 6.404267 |

Totals96 games/14,885 positions/67 terminal games/8157 supported rows,
768 optimizer updates/49,152 sampled training rows/954,682 NN evaluations.
Inferred positions per self-play wall second1250.31/1244.23/1296.95; these include
concurrent actors and their batching, not isolated backend speed.
Generation1 wall295.552876s,parentCPU32.735905s,childCPU838.365110s,
peak parent RSS1405920KiB. Resume invocation wall612.140515s,parentCPU122.660254s,
childCPU1567.652662s,peak parent RSS3934328KiB. Sum907.693391s wall,
155.396159s parentCPU/2406.017772s childCPU. Existing4CPU/16GiB, no new paid resources.
These are two invocation counters, not total project compute or GPU-hours.
Validation-panel preparation overlapped generation1 and compiler setup late
generation3; there is no isolated phase speed comparison. Auxiliary observers'
CPU is outside these pipeline counters. Actors and learner remain eager.

## Diagnostics, support and integrity

Archived teacher validation is stop-only, never training targets. Frozen20,000-row
validation cache SHA
`33fbb59df6b400819ccacc4393c660c8ff2de1c5e8a6328736ef962c8bdb7039`;
cache is derived, not a model/replay checkpoint. PolicyCE2.632099→2.722922,
inside registered2.782099 ceiling but worse. WDLCE0.654349/QMAE0.336883 remain
exactly equal; every frozen parameter unchanged. Known replay fractions45.77%
atgeneration1 and66.65% atgeneration3 passed25% minimum.

All three held-out self-play games hit240plies, giving **zero supported value
rows** in that set. Its displayed value loss0 is masked absence, not perfect
prediction. Generation1 raw shard confirms zero; generation2/3 additionally log
`rolling_validation_terminal_rows=0`. The loss on terminal rows displayed during
policy-only training does not update the frozen value. This phase is limited
policy iteration, not complete self-trained value learning.

Independent shard/checksum/source/opening/history/selected-action/outcome audit
validated96 games/14,885 rows/8157 terminal-supported rows and all capped outcomes.
No16-ply held-out root overlap. Embedded actor book SHA
`019c5b4a49ac287ca03effb65b33ed97daa2e702ddd4a6972491969e27912b24`;
arena book SHA
`85564bdfe43a7a2e4ed37f938cc3223038a8c76147e1bd84d6c478d1adfab06e`.
Arena24 components exclude teacher train/validation, but reuse prior UFUK arena
families with new16-ply roots. Uniform legal suffixes, no quality filtering;
development/stress suite, not independent families or public Elo.

## Fixed arena, uncertainty and source correction

Neural16sim/Gumbel0/one thread;Stockfish19/Threads1/Hash16MiB/requested32nodes.
Same24 roots/both colours/seed20261006/240plies;900s limit per48-game arm.
All original144 games terminated; zero capped outcomes. Independent legality,
endings, score and root audit covers all original12,010 plies and their exact
verification copies.

| Arm | Wins/draws/losses | Score | Pair bootstrap95% | Original wall s / NN evaluations |
| --- | --- | ---: | --- | --- |
| Final vs bootstrap | 5/34/9 | 0.458333 | [0.385417,0.531250] | 169.909013 / 86063 |
| Final vs SF32 | 1/18/29 | 0.208333 | [0.125000,0.291667] | 40.238204 / 18854 |
| Bootstrap vs SF32 | 2/20/26 | 0.250000 | [0.166667,0.343750] | 40.888444 / 19159 |

Final-control SF score delta−0.041667,paired bootstrap[−0.135417,+0.062500].
No improvement proof. Direct primary conservative Hoeffding[0.181112,0.735555].
Automatic `claim_draw=True` includes claims available by an announced next move.
Requested SF nodes and neural simulations are unequal compute; actual SF nodes
are not recorded by this arena. These results remain far from Stockfish strength.

The original CLI sequence declared source`43376c3`, but an optional default-off
compiler adapter was edited in the shared tree during play. This weakened source
provenance even though default eager math remained identical. Original outputs
and dirty patch are preserved; historical source fields are not rewritten.
Following [the preregistered correction](UFUK-arena-provenance-verification-20261003.md),
all three arms were repeated in a clean detached
`43376c35d74a2340773f83a623135e4e0480dc9e` worktree with explicit pinnedPYTHONPATH,
same frozen weights/roots/seeds/budgets. **All144 complete move histories, colours,
endings/scores and NN counts exactly matched**; separately all legal. Copies took
175.990089/40.800817/45.041098s. They are source-integrity verification, not144
additional independent strength samples, and no better score is selected.
The first preflight invoked before the shard audit existed refused before playing;
preserved failure log, no game data. Both sequencing/provenance errors are recorded.

## Decision and reproducible evidence

Frozen value alone was insufficient to make this short policy iteration improve
strength. This does not establish a sole cause. Do not continue this run merely
to obtain a passing score; next measure own-search target quality and residual
value error under [a new fixed diagnostic](UFUK-search-signal-preregistration-20261003.md).
Teacher-bootstrap development improvement remains distinct from failed self-learning.

[Exact text evidence](UFUK-selfplay-evidence-20261003.json) preserves command
receipts, two invocation counters, checkpoints'JSON manifests, retention/shard
audits, both arena sets, dirty patch and verification/audit scripts with SHA-256.
Large model/optimizer/replay bytes remain local and included in the artifact
inventory; Release transport400 prevents claiming a completed remote binary backup.
Linux MLX CPU training/parity are verified elsewhere; Apple Metal/CUDA untested.
Full suite at`0a41a05` (integrated code`c1fb105`),489 passed45.61s,zero skips;
exact stdout preserved in the text evidence. This is not inferred from arenas.
