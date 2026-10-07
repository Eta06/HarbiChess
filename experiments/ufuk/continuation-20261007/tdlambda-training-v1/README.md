# Dormant TD(lambda) fallback adapter

This directory contains a separately versioned target converter, CPU learner,
native proof runner, and arena value adapter for a possible second-stage target
experiment. It has not been run against collection events or connected to the
learner. Use it only if the primary own-Q experiment fails and root opens a
separately registered run.

## Frozen method

The fixed coefficient is lambda = 0.5. Targets are represented in White's
perspective during recursion and converted back to the mover at each row.
Completed games use only rule-replayed own terminal WDL. Capped and row-budget
prefixes use the last logged **pre-action** own search Q as the bootstrap;
there is no invented evaluation of the board after its last recorded move.
Protected games are discarded as a whole. This changes only the target method:
it does not add model/search calls or consume Stockfish labels.

The reported collection receipts show why this is a fallback rather than an
assumed improvement. Each seed has 1,024 training rows. Seed 20262905 has 157
rows from completed terminal games and 867 eligible UNKNOWN rows; seed 20262906
has 97 completed rows and 927 eligible UNKNOWN rows. Thus most targets remain
search-Q bootstraps, not terminal-return labels.

## Input and output boundary

The `prepare.py` API is a pure in-memory verifier, and its CLI reads only the
paths named on its arguments. It requires a root-registered
`own-nnue-tdlambda-target-build-seal-v1` binding the existing Q-v2 dataset,
converter provenance/result, collection receipt, and complete event-log bytes.
It replays every actual move, verifies mover/FEN/history and terminal result,
checks whole-game protected exclusion and the converter's row trace, then emits
new schema names. It does not invoke a producer, model, chess engine, or
optimizer.

The separate admission contract requires:

- dataset `own-kingbucket-tdlambda-training-data-v1`, 1,024 unique rows;
- provenance `NNUE-own1024-tdlambda-target-provenance-v1` and lambda 0.5;
- phase `own-learning-tdlambda-v1`;
- native schema `own-kingbucket-nnue16-tdlambda-full-native-cpu-v1`;
- named-parent weights-only initialization, fresh Adam, accepted-step zero;
- no legacy native resume.

The contract builder binds separate ROOT registrations for a 600-second
conversion, a 600-second proof, and a 1,800-second fresh fit under the existing
operator ceiling. Proof runs 8 updates whole and 4+4 through a fresh process; six
fresh-process loads cover steps `[0,8,0,4,4,8]`. Fresh fit is admitted only with
that exact same-seed, same-data, same-parent proof, then runs 64 fixed updates
with strict `[0,64]` loads. Native state includes model, baseline, Adam, global
Python/Torch RNGs, and private sampler state. Q-only native files are rejected.

`arena_adapter.py` checks the TD(lambda) contract, proof, fit, source and
checkpoint hashes, then exposes the unchanged original NNUE evaluator as the
arena value callback. It does not alter search. Root must still run the
registered profile and game evaluation before making a strength claim.

## Synthetic checks

Run from this folder with the project environment:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/pytest -q -p no:cacheprovider tests
/workspace/HarbiChess/.venv/bin/ruff check --config /workspace/HarbiChess/pyproject.toml source tests
```

Tests use tiny synthetic chess traces, optimizer batches, and contract fixtures.
No actual collection data is opened by this test suite. `prepare.py --help`,
`contract_builder.py --help`, `train.py --help`, and `prove.py --help` are safe
command discovery; actual event conversion, proof, and fit have not been run.
