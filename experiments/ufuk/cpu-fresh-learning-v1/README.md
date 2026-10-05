# Fresh v2 QSEARCH additive own-MC training

Registered CPU training implementation. Actual model fitting/native qualification is admitted only after the fixed independent data checks; it is not a strength result. The input producer is the explicit v2 journal with stored frozen-e8
WDL anchors; the v1 trainer/decoder remains unchanged.

## Input and target contract

`train.py` checks the clean pinned source, exact v2 journal SHA, actor-config
SHA, journal helper SHA, feature helper SHA, frozen-e8 weight SHA, protocol,
training seed, original deadline, and same-seed data pairing. It calls the immutable actual production
`journal_v2.convert_verified`, which independently replays the legal full
history and returns current-state 840 features, mover-perspective terminal
W/D/L, aligned e8 anchor probabilities, game groups, original source IDs, and
a v2 receipt. Capped/active UNKNOWN rows never enter training. The receipt
records the rows excluded. QSEARCH score is not a label.

Admission requires at least 1,024 eligible known rows and 16 distinct complete
trajectories after exclusions, plus nonempty train and validation groups. The
split key is SHA-256 of the exact root FEN, prefix moves, and realized actor
moves; the last byte modulo five assigns bucket zero to validation. Exact
trajectory duplicates are grouped and only one copy contributes rows. Since
all roots are ordinary START positions, this is generated-trajectory-disjoint
internal validation, not root/source-family/state-disjoint validation. Formal
virgin-book strength tests remain the independent gate.

The protocol binds the exact sorted canonical FEN4 protected-position keys
and the exclusion-book SHA from the immutable actor config. The converter
replays each complete game and excludes the entire game if any pre- or
post-action position matches a protected key. It records the exclusion count and a digest of the per-game exclusion ledger.
Unknown and capped games remain unlabeled and are excluded. The data hash
covers features, labels, anchors, trajectory groups, and converter/exclusion
receipts.

## Objective and update budget

Only the sparse additive value head is trainable. The fixed e8 base logits
are represented by the recorded probabilities, so the optimized logits are
`log(base_e8_wdl) + residual_logits`. The loss is own terminal-return WDL
cross entropy plus `beta=1` KL from frozen e8 probabilities to the learned
distribution, plus the unchanged SHRINK spatial/material/metadata penalty.
AdamW learning rate is `2e-5`, weight decay zero, global gradient clip five,
batch size 256. Only the e8 value/policy/trunk parameters outside the sparse
head are frozen. The zero residual starts at exact e8 output.

Use a budget of four batch-sample slots per trajectory-disjoint training row, not
a per-position repeat cap: game-uniform then position-uniform sampling gives
different expected per-row counts for short/long games. The fixed update
count is `min(1024, floor(4 * train_rows / 256))`. This caps total
presentations and avoids pretending every row receives exactly four samples.
Root must lock the admission, protected-position list, trajectory split, and update count before any
strength result is observed.

## Native checkpoint and qualification

Native schema `fresh-qsearch-additive-own-mc-native-v2` binds the full e8
weight SHA, source, trainer/protocol/feature/journal/anchor helpers, exact
actor-config and journal SHAs, converter receipt, dataset and split hashes,
seed, objective, and original deadline. It stores the sparse-head state,
AdamW state, accepted counter, all CPU RNG states, and sampler RNG. The large
frozen e8 base is reconstructed from the same SHA-bound input; the checkpoint
does not duplicate it. Changed input is rejected. No old fullgame dataset is
silently resumed; no older Adam is imported. This is fresh Adam on the fresh
v2 dataset.

`qualify.py` is a root-invoked 8-update infrastructure proof on the actual
v2 journal. It runs whole eight updates versus pause at four and resume to
eight, compares native payloads, then starts six strict no-update loads for
steps 0/4/8 from both runs. This proves native/load determinism only, not
learning or strength. Actual runs should use the original preregistered
deadline and root-owned process/RAM/disk guard.

The trainer uses CPU with one Torch/inter-op thread and a 256 MiB free-disk
floor. Dense arrays are memory-only; journals remain immutable source files.
The output does not save complete copies of e8 in every native checkpoint.
The final full portable candidate is written only after all derived updates;
an eight-step proof result is marked `native-proof-not-candidate`.

## Tests run

`PYTHONPATH=/workspace/HarbiChess/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/workspace/HarbiChess/.venv/bin/python -m pytest -q test_train.py` uses synthetic completed legal mate trajectories plus one cap/UNKNOWN game. It checks the
actual v2 conversion path, mover labels, unknown exclusion, trajectory split,
protected-position checks, changed journal rejection, and zero-residual equality between a tiny
`TorchChessNetwork.masked_policy_value` additive forward and the trainer's
fast sparse critic, including finite nonzero value gradients. These tests do
not load actual e8 weights, run QSEARCH actor inference, exercise the native
CLI over real journals, or provide strength evidence. Root must run the
pre-result actual-journal restart qualification before admitting a production
fit.
