# UFUK: Q-range floor mechanism test before new search runs

Width failed its registered native learning gate; widening is not promoted.
Hypothesis: min/max normalization overamplifies small inaccurate shallow Q
differences. A larger absolute WDL-Q range floor may improve selection and soft
policy targets. This is an upstream-related Gumbel ablation, not a new invention.
[Primary mechanism audit](../research/UFUK-search-scaling-audit-20261004.md).

Frozen original18f native model, existing128 full-history positions/16families
and3433all-legal independent Stockfish references from the native target panel.
No new engine labels, no gradients, no new models or paid allocation.
All three arms16sim/max4/Gumbel0/value_scale0.1/maxvisit_init50/eagerCPU1/B1.
Only Q denominator floor changes: historical1e-8 control versus0.05 and0.20.
Original public search and defaults unchanged; a separate experimental subclass
must reproduce default output and preserve legal moves/visit/sign semantics.
Floor>=range matches the original scale when equal, damps small ranges when
greater; terminal outcomes remain exact. Seed20261018, alternate three-arm order
by position index;300s total diagnostic wall and7500NN-call cap, durable full rows.
Stop mismatch/nonfinite/budget; preserve incomplete evidence, no lucky extension.

Endpoints: selected-move expected score and soft-target expected score versus
historical floor, weighted equally by the16whole families (8positions each).
Report individual16family effects and10000bootstrap resamples/seed20261018.
Four candidate/endpoint contrasts: Bonferroni-adjusted two-sided98.75% intervals
(quantiles0.00625/0.99375), nominal95% also explicitly diagnostic; conservative
Hoeffding family bounds separately. Require selected mean gain>=0.01 with
adjusted lower>0 AND soft-target mean>=0.01 with adjusted lower>0 for that arm.
If both qualify, select0.05 by preregistered closest-to-default preference,
not highest measured effect. No arena if neither qualifies. Reused development
positions/method-selection history limit inference; bootstrap is conditional,
not independent confirmatory significance or Elo.

Before diagnostics freeze24new40ply game roots by four uniform-legal suffix
moves from the unused width36ply book; seed20261018, native/prior-complete-arena
current states excluded, no engine/model filtering. Same development families.
If qualified, selected floor versus original at same weights/search budget and
each versusSF32,48colour-matched games per arm, max240plies/900s per arm/seed18,
CPU1/eagerB1. SF19/Threads1/Hash16MiB/32requested with per-ply actualnodes.
Strength gate score>0.60,24pair bootstrap95%lower>0.50,caps<=0.10; count caps0.5
only diagnostic, never outcome training labels. Record complete wall, NNcalls,
durable moves, median/p95 move time and pairedSF uncertainty. Unequal neural
simulations/engine nodes, not equal wall/compute engine superiority.

Even a search-strength pass is not successful model learning. It can justify a
separately preregistered fresh final-versus-initial self-learning experiment using
the same search in both arms. Main Stockfish/AlphaZero-level strong, fast and
reliable self-learning goal remains unmet. Original failures remain failed.
