# Read-only review of fresh QSEARCH producer and residual WDL requirement

No active helper, input, model or process was changed. No NN inference, training, match, engine or network job was launched. Rule/metadata checks used one CPU affinity and wrote only small proof files.

## Actual recovery evidence

`published-prefix-proof.json` protects a complete-line snapshot of the live log. Both original raw game_end strings match exactly. The interrupted game's ten moves match the first ten recovered moves, with full-history legal replay. Original group and whole first/deadline match parent receipts. New executor SHA is separately bound; original parent executor remains distinct.

`chronological-prefix-packets.json` independently checks the published completed recovered game's ten prefix packets: same move/ply, correct mover/legal-root count, finite counters/values/timing and honest measurement-origin tags. They total 5,120 NN nodes and 4,745 evaluations. This checks recorded packets; it does not independently recompute the network/search. Old partial latency/counters remain unavailable. Group completion, all 160 games and numerical gates need the final audit.

## Fresh producer: useful behavior and remaining qualification limits

- This is a fresh actor/RNG schema. Search selects a deterministic value-search action; played action is .95 selected + .05 uniform legal. Uniform exploration can select the same move, and stored mu correctly includes both branches. It is not the network policy softmax or PPO behavior, and search_value is not an outcome target.
- Full legal root prefix and active played moves survive a pause. Whole9 vs pause4/fresh-process resume9 gzip/RNG equality is proven only with synthetic fixtures. Actual real frozen model/source/runtime CLI and chronological NN packet qualification remain outstanding.
- Complete outcomes are recomputed with claim_draw=True and terminal precedence at total-ply400. Cap games are UNKNOWN and unfinished epoch tails are excluded, not draws. SHRINK labels use mover POV and game-uniform groups, preserving original source IDs. This is conditional-on-completion data; exclusion and game balancing do not remove that sampling bias.
- Snapshot model hash/config and original deadline are fixed across CLI segments. There is no optimizer/learner checkpoint or old online-native resume claim. A changed model must start a newly registered epoch; it cannot silently continue an active old-model tail.
- Production still needs root-owned periodic checkpoint scheduling. Current `produce.py` publishes only after its requested segment; deadline/process failure mid-segment loses that unsaved tail. Use bounded consecutive milestones under the same original deadline, retaining each immutable checkpoint; do not claim uninterrupted recovery of lost actions or reset budgets.
- Replay deliberately checks saved search counters structurally, not the internal search tree or neural values. Its counter bounds permit depth0/nodes0 synthetic fixtures; actual qualification should additionally require original BudgetSearch full-root accounting (root_actions+1 <= nodes, positive completed depth, value within terminal range) and chronological packet reproduction.
- In-memory Actor keeps references to caller config/state; the CLI does not mutate them, but a future in-process caller could mutate config after construction. Before using it in a persistent controller, deep-copy/freeze those objects or check original config digest at every advance. A changed serialized config is rejected on replay; this is still avoidable in-process misuse.
- Torch version is checked in real CLI. Python/python-chess/NumPy versions and CPU/thread/runtime identities should also be registered before actual qualification, since source SHA alone does not pin library semantics. Tests and journal checksum are not substitute hardware/NN evidence.

## Residual full-WDL adapter is mandatory before a residual model arena

Current arena `value.py` uses `model.value_sparse_head` alone whenever value_sparse exists. It intentionally evaluates the replacement SHRINK critic. It will NOT automatically evaluate `full frozen e8 logits + learned residual logits`; therefore feeding a residual-only sparse artifact to the current arena would omit its anchor. No residual runtime artifact was inspected or qualified in this review.

Prospectively specify a dedicated SHA-bound residual evaluator. It should obtain original e8's complete three WDL logits through the real full network value forward on the same complete history, add the precisely registered three-logit residual, apply one softmax, then use win-minus-loss for mover search. Do not add scalar W-L values and call that equivalent to logit residual WDL. Explicitly prove zero residual reproduces e8 full WDL on diverse white/black histories (castling, EP, repetition, halfmove metadata), and nonzero residual matches the training expression. Frozen e8 and residual source/model hashes, coefficient/temperature, both mover labels, terminal handling and portable parity must be bound.

The two-forward residual evaluator may cost more per leaf than current sparse replacement. Record primitive NN forwards/evaluation and actual move latency; equal512 search-node limits alone do not establish equal compute or the unchanged latency gate. Training-only loss/zero-residual parity cannot establish strength. Fixed final model, matched e8 and zero controls, original direct/SF/cap/uncertainty/latency gates remain required without outcome-driven changes.
