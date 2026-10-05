# QSEARCH v2 with frozen e8 WDL anchors

Scratch proposal only. No NN self-play actor, full dataset conversion, model
fit, or strength match was run. `journal.py` v1 under
`cpu-qsearch-selfplay-proposal/` remains untouched. This folder contains a
separate `journal_v2.py` schema, `produce_v2.py`, a small anchor adapter, and
CPU-only synthetic tests. V2 does not reinterpret or auto-upgrade a v1
checkpoint.

## What v2 records

Before each actual actor action, the actor passes a copy of the full-history
board (including the move stack) to a separately identified frozen e8 value
helper and records its mover-perspective normalized W/D/L probabilities in
`e8_anchor_wdl`. This is an independent baseline inference, not the QSEARCH
`search_value`, not an external teacher, and not a terminal label. The v2
config binds anchor model and helper SHA values, anchor convention, actor
model/helper/source identities, search settings, rootbook, seed, and original
deadline. Replay checks every recorded anchor exists and is finite,
nonnegative, three-way normalized, and aligned with the replayed pre-action
position. It cannot independently recompute e8 inference; production audit
must replay selected rows with the pinned e8 model/helper and compare outputs.

The current `cpu-budget-search-v1/value.py::NeuralValue.logits` helper returns
only `value_sparse_head(inputs)` whenever a model has any sparse-value schema.
That omits inherited base logits for an additive schema2 candidate. The plain
e8 anchor path is unaffected, but before an additive candidate is used as an
actor model, pin a new helper that runs the full base-plus-residual forward
path and test parity with the deployed backend.

The QSEARCH actor remains fixed: one CPU actor; 512 nodes, qdepth 2, max depth
8; 0.95 selected move plus 0.05 uniform legal exploration; 400-ply cap;
8,192 action maximum per seed. Actor action/mu, search choice, search value,
and frozen base anchor stay distinct in each transition. Only completed
terminal games train; cap/active games remain UNKNOWN. V1's 2 MiB artifact
limit remains unchanged; v2 prospectively raises its own artifact-directory
ceiling to 16 MiB, without deleting earlier snapshots. The owner must register
an original deadline and keep at least 256 MiB free.

## Additive own-return objective

For each known pre-action row, convert the stored base WDL probabilities to
log probabilities and add the trainable sparse residual logits. Optimize
terminal own-game W/D/L cross entropy plus KL from the frozen e8 base
distribution to the learned distribution. The policy, trunk, and search
procedure stay frozen. `objective.py` implements that composition and the
tests verify that a zero residual reproduces exactly the base probabilities.
Targets are mover-perspective terminal returns; search scores are not labels.

The fixed presentation budget is four batch slots per eligible training row,
with `min(1024, floor(4 * training_rows / 256))` updates of batch size256.
Game-uniform sampling does not cap an individual row at four draws. Do not run unless the terminal-data
count and generated-trajectory-disjoint internal validation split meet a preregistered minimum; a
tiny dataset should lead to no fit, not repeated presentations. Root still
chooses/fixes the exact minimum, optimizer settings, anchor KL weight, and
candidate initialization before production. Two proposed seeds are
20262805/20262806, with ordinary initial-position roots and any reserved proof
roots excluded from training.

The new data generation requires an explicit native v2 contract binding all
v2 journals, e8 weights/helper, actor model/helper/source, feature helper,
source partitions, training seed, and initial candidate. It must store model,
optimizer, update cursor, Python/Torch/sampler RNG, data SHA, per-journal
receipts, and the original deadline. A v1 native cannot be treated as a v2
full resume. If root elects to reuse the old residual candidate's Adam, that
must be recorded as a new data-generation continuation with exact parent
native SHA and tested restoration; otherwise start fresh Adam. No actor,
search RNG, or old dataset resume is implied by loading identical weights.

## Cost and acceptance

The existing QSEARCH qualification measured 24 512-node moves over four roots
per seed at about 8.8 seconds for the listed profile. A linear extrapolation
puts 8,192 moves near 50 minutes per seed before v2's extra anchor inference,
replay, training, or audit. This is only a planning estimate. First measure a
short representative actor profile including anchor capture, and verify
throughput on the actual full-history data path under CPU1. The extra anchor
call could dominate if not reused; it must not silently be omitted to preserve
speed. A safe optimization is to have the same immutable model return the
full WDL logits from the already cached root NN evaluation, but only after a
versioned cache contract and exact equivalence tests. V1 search root visits
must not be changed for that optimization.

Training status is not strength. The unchanged two-seed SF512NODES/96-game
gate is the only strength decision, comparing fixed e8 and learned candidates
under the same deployed search and evaluation setup. Do not select seeds,
book, optimizer, checkpoint, or reported candidate based on interim match
results.

## Tests run

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/HarbiChess/.venv/bin/python
-m pytest -q test_journal_v2.py` uses a tiny scripted legal checkmate and fake
normalized anchor. It checks full-history pre-action capture, mover targets,
search-score separation, replay corruption rejection, v1/v2 schema separation,
deterministic whole versus pause/resume journal bytes, immutable config,
zero-residual anchor equivalence, and invalid anchor rejection. These are
synthetic rule/serialization/objective checks, not actual e8 inference or
NN-qualified collection.
