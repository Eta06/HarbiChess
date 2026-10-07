# ROOT-only MC phase handoff

This directory is an integration layer for the frozen `../source` producer. It does not select roots, run collection, train, or assign a clock. ROOT must register each proof/fitting clock and retain the existing collection registration unchanged.

## Frozen MC identities

- Collection: `own-nnue-closed-terminal-collection-registration-v1` / receipt `own-nnue-closed-terminal-collection-receipt-v1`.
- Converter output: `own-kingbucket-closed-terminal-training-data-v1`, provenance `NNUE-own1024-closed-terminal-data-provenance-v1`.
- Training phase: `own-closed-terminal-learning-v1`; native `own-kingbucket-nnue16-closed-terminal-full-native-cpu-v1`.
- Target: `frozen-parent-own-search-closed-terminal-wdl-v1`; only real terminal W/D/L from complete unprotected games trains. Capped games stay UNKNOWN and are excluded.
- Proof/fit status: `PASS-closed-terminal-fixed-phase-and-fresh-native-loads-not-strength`.
- Arena hook: `arena_adapter.admit_closed_terminal_child(record)`; this admits a value endpoint only. It does not qualify search or strength.

Use the same-seed original teacher candidate as a weights-only initializer; preserve the original frozen-parent contract as a provenance input. The own phase gets a fresh optimizer and RNG state. No old ownQ/TD native or audit result can be reused as an MC proof.

## Execution order

1. ROOT freezes the two collection registrations with the existing source/pool/protected alias SHAs and one original collection clock per seed. Run `source/run_collection.py --registration <registered.json>`. Keep the producer receipt, complete `events.jsonl`, and every `search-aliases-*.bin` file together.
2. Construct the converter seal with `seal_factory.make_conversion_spec(registration_path, receipt_path, first=..., deadline=..., core_repo=...)`; persist it publish-once. The supplied `first` and `deadline` must be the separately ROOT-registered converter clock. Run `source/convert.py --seal <seal.json> --output /dev/shm/...` under that same clock. The converter validates every event, full history, game result, UNKNOWN tail, selected row, and contiguous sidecar span before emitting the 1,024-row dataset/provenance.
3. Run `integration/audit_six.py` through the ROOT audit harness on the same immutable registration, receipt, events, chunks, parent, and original search helper. It performs full converter replay plus fresh search recomputation for receipt rows `[0, 204, 409, 614, 819, 1023]`, and verifies every evaluator alias for those six rows. Required result schema/status: `NNUE-own-closed-terminal-six-search-audit-v1` / `PASS-six-actual-chronological-terminal-eligible-search-packets-and-complete-alias-traces`. The audit clock schema is `NNUE-own-closed-terminal-six-search-audit-clock-v1`; it must pin the spec, registration, receipt, events, this helper, and `source/convert.py`, plus one original first/deadline and CPU core. Preserve its original audit clock, inputs, and source SHAs. This is six packet checks; full event and terminal-label validation is separately supplied by the converter replay.
4. After conversion and audit pass, ROOT writes a proof build seal with `seal_factory.make_seal(mode="proof", ...)`, using exact converter dataset/provenance, same-seed parent/contract, source closure, and registered proof clock. Run `source/contract_builder.py` then `source/prove.py` under the proof registration. Require whole 8 steps, pause at 4, fresh-process resume to 8, payload identity, and fresh strict loads at steps `0, 4, 8` for all required payloads (six payload checks total). No model outcome selects a schedule.
5. Only after proof PASS, ROOT writes a separate fresh-fit build seal with `mode="fresh-fit"` and pinned proof result/contract; build the contract and run `source/train.py` under the single registered fresh-fit clock for fixed 64 updates. Require fresh strict loads at 0 and 64 and a changed model. There is no loss-based selection.
6. Build one typed `closed-terminal-mc-v1` variant record per seed for the known160 adapter. Bind contract, proof and fit registrations/results, all native payload refs, candidate and parent, conversion provenance, collection receipt/registration, events/chunks, and this MC audit result/clock. The adapter must rerun the frozen converter from the bound inputs and reject every non-MC audit schema/status. Keep the established E8 and parent strength comparisons and gates unchanged.

No step may extend an exhausted collection/proof/fit/audit clock or reuse a clock after a failed attempt. All publish-once run outputs belong under `/dev/shm`; keep source receipts and failed attempts intact.

## Seal construction API

`seal_factory.make_conversion_spec(...)` creates only a converter input seal from a completed passing receipt; it does not stamp a time. `seal_factory.make_seal(...)` creates a proof or fresh-fit contract-build seal from already-produced converter artifacts. Both accept explicit root-supplied clock values and reject absent refs. `write_once(path, obj)` refuses replacement. The normal frozen helpers remain authoritative for admission and native verification.

## Local integration checks

`/workspace/HarbiChess/.venv/bin/pytest -q integration/test_audit_six.py` uses synthetic packets only. It does not call the real model, search, converter data, or optimizer. The original frozen source tests remain separate.

The typed per-seed child record is produced by `seal_factory.make_child_record(...)`. It contains SHA refs for contract, proof contract, dataset, target provenance, candidate, proof/fit results and registrations, collection receipt/registration, events; it derives `initial` and `native` from the fit result's actual fresh-process step-0 and step-64 payloads and binds the original proof/fit build-seal files. Supply that record under `children[seed]` with global `target_variant="closed-terminal-mc-v1"`; set `variant_helpers` to the pinned MC `model.py`, `native.py`, `contract.py`, `convert.py`, and `contract_builder.py`. The separate audit set is made by `seal_factory.make_audit_set([{seed, result, clock}, ...])`; it only accepts both fixed seeds and the MC audit schema/status. Keep the parent candidate in the protocol's existing `teachers[seed]` record, and the selected child candidate in `models[seed].learned`.

`make_child_record` requires `variant_helpers` as a name-to-path map for exactly those five frozen MC files. It checks each helper SHA against the resulting contract closure; the returned child includes these named refs, so the adapter does not infer code identity from a directory.
