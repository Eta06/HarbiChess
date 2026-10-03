# AYNA immutable rule-query cache

Before implementation/measurement. Spawn improved the frozen8-game throughput
1.766x (512positions in14.811s versus8.386s) while using3.6CPU including startup.
Rules repeatedly compute full legal moves, FEN views and claimable outcomes for
immutable full-history states during search. Claim checking can replay history
and legal continuations. This may still dominate CPU; no profiler attribution yet.

Hypothesis: bounded per-thread memoization of immutable legal/view/outcome facts
can reduce redundant work without changing rules, draw-claim convention, encoding,
model/search arithmetic or replay/checkpoint schemas. Cache key is full root FEN
plus all moves, and claim_draw is separately keyed; current FEN alone is unsafe.
Cache None outcomes correctly, retain thread isolation, max512states, no mutable
borrowed boards exposed as facts. Preserve current claim_draw=True semantics,
including claim by announced next move, and document that arena convention.

Correctness: compare cache hits to independent python-chess for both claim flags,
same current board/different history, castling/en-passant/promotion, terminal and
ongoing roots. Existing rules/history/search/learner tests must stay passing.

Speed control: isolated source845ae7a (before cache) versus optimized source,
same frozen bootstrap weights, eight games/64plies/16sim/seed20261004, thread mode,
120s ceiling each, exact prior actor benchmark. No concurrent training/arena.
Success >=15% unique generated-position throughput and matching complete legal
game histories/targets; report if bitwise replay records differ. No strength gain
claimed from caching. Startup and encoder/rule/search/inference time all included.
Existing learning processes use already imported code; no mid-run rule mutation.
