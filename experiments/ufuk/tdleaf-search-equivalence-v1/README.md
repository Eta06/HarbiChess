# TDLeaf search instrumentation qualification (source only)

This is a separate ROOT-run helper. It has not loaded a production parent or
run a real search. It compares the pinned de53 baseline search with the
instrumented PV search on the first 24 eligible TRAIN roots from the exact
registered CLASSIC4096 pool. The immutable pool and protected-alias file are
inputs; the helper does not regenerate, reorder by outcomes, or substitute a
procedural bank.

For each root it replays and retains the complete source history. It runs both
searches at 8192 nodes, qdepth 2, max depth 8, alternating order by fixed row
index. It compares selected move, value, node/evaluation counts, completed
depth, root actions, and every ordered evaluator input: board FEN, full UCI
move stack, and returned scalar in exact hexadecimal form. It also verifies
the captured legal PV, endpoint rule status, and all protected aliases. No
weights change, game collection, labels, or optimizer updates occur. The
paired aggregate wall ratio must be at most 1.10. The recorded ancestral
prefix intersections are diagnostics; only current roots, actual evaluator
inputs, and root-to-PV paths are in the protected scope.

ROOT must write a registration using schema
`tdleaf-search-equivalence-registration-v1`, with an observed first/deadline
no longer than 600 seconds, unchanged source/pool/protected hashes, the exact
parent admission packet, core commit, CPU core, and a new `/dev/shm` output
path. The `source` object has refs `baseline_search`, `pv_search`, and
`parent_runner`; `baseline_search` must equal the admitted parent search helper.
`parent_runner` must name the pinned TDLeaf runner containing the current-parent
factory. All production refs remain placeholders in no template; only ROOT can
register and execute the actual profile.

Synthetic test command:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python -m pytest -q -p no:cacheprovider test_qualify.py
```
