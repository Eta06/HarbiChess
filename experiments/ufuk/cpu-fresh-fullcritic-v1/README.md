# Full value-head own-search proposal

Scratch proposal only. No actor, E8 inference pass, match, optimizer update, model fit, or production-source edit was run here. The exact mirrored E8 weights were read to verify model specification and the trainable mask. The expected parent path is a third controlled capacity/target experiment after ordinary own-MC and sparse full-history variants; this is not evidence that it will improve strength.

## Model change

Start from the exact frozen E8 pairwise model. Keep the shared stem, residual blocks, all policy modules, and all other parameters bitwise frozen. Train only existing active value modules: `value_conv`, `value_hidden`, `value_output`, `invariant_value_linear`, the `value_tower_*` family, and `global_value_*`. Do not attach a sparse head. The model begins as the exact E8 function and uses a fresh AdamW optimizer; it is weights-only initialization, not a resume from an older optimizer.

The exact mirrored E8 artifact is pairwise with 16 trunk channels, 2 value channels, 32 value-hidden units, and invariant `(channels=16, blocks=2, hidden=32)`. The selected existing value modules contain 31,290 trainable scalars; the trainable-name digest is recorded in `protocol-template.json`. Float32 parameter storage plus Adam first/second moments alone is 375,480 bytes. This method has a separately versioned 2 MiB native/candidate file ceiling. The old sparse method's 512 KiB cap remains unchanged.

## Data and targets

Use the frozen common trainer's exact 12-field `prepare_fresh` result once. Reuse its protected-position exclusions, duplicate-trajectory handling, train/validation groups, and row indices. Then re-encode the same replayed full histories to `(8, 8, 104)` planes and independently verify that its 840 invariant features, mover labels, and stored E8 anchors match byte-for-byte. Attach search targets using the sibling's pinned own-search extractor/binder (`own_search_consistency.py`, SHA in the protocol). UNKNOWN caps and open tails stay excluded.

Each row has (1) terminal WDL from the actual mover's perspective and (2) the recorded selected-best-move root score from that same pre-action root mover. Clamp only the score target to `[-1, 1]`; do not negate it and do not replace it with the outcome of an exploration action. The tests include a black-to-move position whose recorded `+0.25` search score is preserved while the completed game labels that row as a loss.

## Fixed objective and limits

The proposed loss is `0.5 * own-terminal CE + 0.5 * MSE(p(win)-p(loss), clamped own-search score) + 1.0 * KL(frozen E8 WDL || current WDL)`. Search targets and base anchors are detached. AdamW learning rate is `2e-5`, weight decay zero, gradient clip five, batch 256, and at most four batch-sample slots per eligible train row. Update count is `min(1024, floor(4 * eligible_train_rows / 256))`. The architecture-specific sparse SHRINK penalty is zero because it has no defined full-head equivalent. That makes this a different regularizer from the sparse experiment, so a later causal capacity-only claim would need a matched functional-regularizer control.

The scratch native helper stores only trainable value tensors, Adam state, accepted count, frozen-parameter digest, sampler RNG, Python/NumPy/Torch CPU RNG, and a caller contract that binds the exact E8 hash/spec, common dataset/search target receipts, and deadline. It rejects a changed contract, mask, or frozen model before restoring. A synthetic whole-two versus pause-one/resume-two unit test round-trips this payload in a fresh model and compares all model tensors, Adam state, and RNG cursors exactly. The standalone CLI now implements the registered 8-update whole versus 4-update pause/resume proof and six strict fresh-process loads at checkpoints 0/4/8. The protocol remains draft until root freezes the final journal SHAs, clocks, source pin, and helper bindings. No actual-journal qualification or fit has been run here.

## Tests

Run with the HarbiChess venv and one CPU thread:

```sh
PYTHONPATH=/workspace/HarbiChess/src:/workspace/work/harbichess/cpu-fresh-additive-anchors-proposal \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/workspace/HarbiChess/.venv/bin/python -m pytest -q test_core.py
```

The tiny tests use a random default pairwise model and one synthetic, legally replayed full game. They check full model forward/backward, value-only gradients, bitwise shared/policy freeze, dense-encoder and 840-feature parity, terminal mover labels, selected-search-score alignment, and fixed loss weights. They do not load the actual E8 artifact, run a fit, validate a full native resume, or measure strength. MLX parity and hardware remain unverified.

## CLI status

`train.py` consumes the exact common 12-field fresh-data helper, pinned v2 journal/anchor helpers, and the sibling own-search extractor. It checks the actor exclusion list, source/model/anchor hashes, data rows, search targets, parameter mask, fixed objective, and fixed original deadline before updating. It emits complete native checkpoints and a candidate only after the frozen update budget. `qualify.py` is an actual-data whole/pause/resume driver; it is prepared but unexecuted. The current template intentionally remains unregistered because each seed's final journal SHA and owner clock must be supplied by root.
