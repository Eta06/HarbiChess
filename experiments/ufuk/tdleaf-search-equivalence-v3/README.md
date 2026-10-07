# TDLeaf search instrumentation qualification v3 (source only)

This separate ROOT-run helper has not loaded a production parent or run a real
search. It uses the exact registered CLASSIC4096 source and first 24 eligible
TRAIN roots, in source order. It does not regenerate, reorder by outcomes, or
substitute a procedural bank.

For each root, it replays full history and runs four paired searches at 8192
nodes, qdepth 2, max depth 8: traced baseline/PV searches and separate
untraced baseline/PV searches. Traced order alternates by root index; the
untraced pair uses the opposite order. Both traced search paths perform the
same observer and evaluator telemetry once per evaluator input. The untraced
pair calls the identical evaluator with no qualifier trace/observer to measure
search instrumentation overhead separately. Both pairs must produce exactly
equal result fields, including `value.hex()` (so signed zero is distinguished),
and both PV-to-baseline timing ratios must be at most 1.10.

The traced comparison retains every ordered evaluator input as exact FEN,
complete UCI move stack, and scalar `float.hex()` packet. It also verifies the
captured legal PV and checks protected aliases on all evaluator/observer
inputs and every position from current root through terminal/static PV leaf.
Historical ancestry intersections remain diagnostics and do not expand the
registered protected scope.

Full query packets use deterministic gzip JSON, one publish-once file per
root. Each packet has a 32 MiB uncompressed ceiling and an 8 MiB compressed
ceiling; all packet files together are capped at 128 MiB. The report is capped
at 8 MiB and holds exact file path, byte length, and SHA-256 references rather
than duplicated query arrays. Every packet is reopened and checked before
the report is published. Truncated, concatenated, or trailing compressed data
fails closed.

The ROOT registration schema is
`tdleaf-search-equivalence-registration-v3`; the result schema is
`tdleaf-search-equivalence-result-v3`. ROOT supplies the observed original
deadline (at most 600 seconds), parent admission packet, clean core commit,
CPU core, frozen source/pool/protected refs, and new `/dev/shm` report and
packet-directory paths. The source refs are baseline search, PV search, and
parent runner; baseline must equal the admitted parent search helper. Only
ROOT can register and execute the real qualification.

Synthetic test command:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python -m pytest -q -p no:cacheprovider test_qualify.py
```
