# Classical own-experience PST residual proposal

This is a scratch implementation proposal. It has run no self-play, search,
fit, match, network request, or arena inspection. It is a standard feature
extension, not a novelty claim.

The model preserves the frozen 18-term human evaluation and its trainable
residual. It adds 224 coefficients initialized to zero: five non-king piece
tables over 8 ranks × 4 horizontally mirrored files, plus separate
middlegame/endgame king tables. All terms are signed from side-to-move
perspective and use vertically oriented squares, so color/rank reflection is
preserved. Zero PST coefficients must produce bit-identical classical18 values.

## Data and objective

`frozen_reference/` is byte-copied from the existing classical18 proposal and
hash-pinned in `inventory.json`. `prepare_pst.py` calls that exact frozen
`learner.prepare` first. It then reconstructs PST features from the same
replayed full histories and asserts identical trajectory keys, split, rows,
original feature columns, prior values, terminal labels, Q labels, and row
counts. Caps/unfinished tails remain unknown. The fresh PST feature columns
are additional inputs, never labels.

Training keeps the existing loss weights (0.75 terminal mover-WDL MSE, 0.25
own-Q512 MSE), Adam constants (lr .01, betas .9/.999, eps 1e-8, batch 256,
clip 5), game-equal then position-equal sampling, and `min(1024,
floor(4*training_rows/256))` updates. Existing 18-term L2 remains .01 times
mean squared theta. New fixed regularizers are .02 times mean squared PST
coefficient and .01 times mean squared differences over 364 fixed adjacent
rank/file edges. These PST regularizers are an additional objective change;
the result is not a pure capacity-only ablation.

The fit contract binds the original protocol, seed, exact config/journal SHA,
the independently frozen classical18 and PST dataset SHAs, terminal/Q targets,
optimizer/update count, and both regularizers. `pst_native.py` has a new schema and stores all 242 parameters,
Adam moments, update cursor, candidate, sampler RNG, and global RNG. It rejects
classical18 native checkpoints. Starting from the prior is a new weights-only
initialization with fresh Adam; no old optimizer/RNG state is carried forward.

`mixed_value.py` has separate roles for E8 (explicit evaluator callback),
human prior, classical18, and PST. Chess rules and search helpers are not
modified. ROOT must review/freeze actual journal/config/deadline hashes and
qualify both seeds whole8/pause4/freshresume8 plus six strict loads before any
fit. If later fitted, strength uses the already fixed search and unchanged
strength gates; training/validation loss alone is not a strength result.

`data-preflight-receipt-launch-ready-v2.json` verifies both exact frozen datasets
against those protocol SHAs. It records no SGD, search, or game execution.

`launch_pst.py` is the unexecuted two-phase launcher. Its proof phase runs each
seed whole to step 8 and as a separate 4-step pause plus fresh-process resume to
8, checks full-state equality, and performs strict native loads at steps 0, 4,
and 8 for both seeds. A fit phase requires the proof receipt SHA, then starts
fresh zero-residual/fresh-Adam fits sequentially under a separately registered
1,800-second clock. It strict-loads step zero and the final native for each
seed. All RAM outputs are under distinct `RAM/<seed>/...` trees so the existing
16 MiB guard is scoped per seed; the launcher also guards the 32 MiB cohort cap.
The phase clocks must be frozen before the proof phase and may not be extended.

Related practice includes classical piece-square tuning, TDLeaf, Texel tuning,
and neural chess evaluation. No originality or strength claim is made.
