# Procedural own-data forensic adapter v3

This stage preserves the original v2 registration and raw receipt. It handles one narrowly documented receipt defect: `run_collection.execute(path)` reuses `path` in the generation-helper loop and later hashes that loop variable into `registration_sha256`. The raw receipt field is therefore the registered `parent_bridge.py` SHA, while the untouched original registration has a different SHA. The source is AST-checked for this exact loop-before-assignment pattern. V3 emits a separate forensic view; it never writes a “corrected” v2 receipt.

Actual metadata-only derivation passed for both seeds:

- 20262905: raw receipt SHA `3bd364…`; original registration SHA `1732eeb…`.
- 20262906: raw receipt SHA `3bd364…`; original registration SHA `745ecd…`.

The view also binds the pre-actor build seal, controller registration, owner process, and the controller’s pinned inputs. The initial six-packet v2 attempts failed before any search inference because they rejected the raw receipt SHA. Preserve those failed attempts; do not reuse their clocks or label them v3 passes.

## ROOT execution order

All paths below are under `/workspace/work/harbichess/continuation-20261007`; run from this stage using `/workspace/HarbiChess/.venv/bin/python` and `PYTHONDONTWRITEBYTECODE=1`.

1. Publish each view once under `/dev/shm/harbichess-continuation-20261007/forensic-v3-SEED/receipt-view.json` using `forensic_receipt.py`. Inputs are `human-random-collect-actual-SEED/collection-registration.json`, the original receipt at `/dev/shm/harbichess-human-randomstarts-ownq-v2/g-1/SEED/receipt.json`, and producer directory `human-randomstarts-own-v2`.
2. ROOT stamps a new original 600-second clock per seed with `prepare_forensic_audit_clock.py`. The clock binds registration, receipt, event JSONL, view, runtime helper SHA, source directory, CPU, and operator end.
3. Run `audit_collection_six_v3.py` once per seed with those exact four inputs, the clock, and a new result path `/dev/shm/harbichess-human-randomstarts-forensic-audit-v3/SEED/result.json`. It independently replays only six preregistered row ordinals and checks each actual search result plus complete alias trace. It records zero new games/rows/optimizer updates.
4. Only after both pass, build the paired manifest with `prepare_forensic_audit_set.py`, binding the v3 audit runtime and validator separately. The manifest validates raw receipt bytes and bad field, registration and view bindings, clocks, results, and the exact six row IDs.
5. ROOT stamps a new conversion clock and per-seed converter seals with `prepare_forensic_conversion_v3.py`, then runs `convert_forensic_v3.py`. Conversion remains independent full replay over all rows and sidecars; six packets are not treated as a substitute for full conversion.
6. Build the new proof/fresh-fit contract with `prepare_forensic_contract_v3.py` and `contracts_forensic_v3.py`. The contract requires the v3 paired audit set and same-seed conversion provenance. The existing learner/native payload format is reused only through an explicit v3 contract; it does not relabel a v2 dataset or receipt.

No v3 six-packet replay, conversion, proof, or optimizer step was run by this stage. The source changes are scratch-only and need ROOT’s review and source pin before execution.
