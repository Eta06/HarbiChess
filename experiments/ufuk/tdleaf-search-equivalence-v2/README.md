# TDLeaf search instrumentation qualification v2 (source only)

This is a separate ROOT-run helper. It has not loaded a production parent or
run a real search. It compares the pinned de53 baseline search with the
instrumented PV search on the first 24 eligible TRAIN roots from the exact
registered CLASSIC4096 pool. It does not regenerate, reorder by outcomes, or
substitute a procedural bank.

For each root it replays and retains the complete source history. It runs both
searches at 8192 nodes, qdepth 2, max depth 8, alternating order by fixed row
index. It compares selected move, exact hexadecimal root value (including
signed zero), node/evaluation counts, completed depth, root actions, and every
ordered evaluator input: board FEN, full UCI move stack, and returned scalar
in exact hexadecimal form. It verifies the captured legal PV, endpoint rule
status, the full root-to-PV path, and every evaluator/observer input against
the pinned protected aliases.

The full ordered query packets are stored as deterministic gzip JSON, one
publish-once file per root under a new `/dev/shm` directory. Each file is
bounded to 8 MiB compressed and uncompressed; aggregate packet files are
bounded to 128 MiB. The small report stores their exact paths, byte lengths,
and SHA-256 digests, then reopens and validates every packet before publishing.
The report itself is bounded to 8 MiB. Truncated, concatenated, or trailing
compressed data fails closed. The report includes no large duplicated query
arrays.

No weights change, game collection, labels, or optimizer updates occur. The
paired aggregate wall ratio must be at most 1.10. Recorded ancestral prefix
intersections are diagnostics; only current roots, actual evaluator inputs,
and current-root-to-PV paths are within the protected scope.

ROOT must write a registration using schema
`tdleaf-search-equivalence-registration-v2`, with an observed first/deadline
no longer than 600 seconds, unchanged source/pool/protected hashes, exact
parent admission packet, core commit, CPU core, and new `/dev/shm` paths for
both the report and the per-root packet directory. The `source` object has
refs `baseline_search`, `pv_search`, and `parent_runner`; baseline search must
equal the admitted parent search helper. `parent_runner` must name the pinned
TDLeaf runner containing the current-parent factory. Only ROOT can register
and execute the actual profile.

Synthetic test command:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python -m pytest -q -p no:cacheprovider test_qualify.py
```
