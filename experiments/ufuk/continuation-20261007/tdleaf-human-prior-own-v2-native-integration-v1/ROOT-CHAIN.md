# TDLeaf v2 native integration derivative (source only)

This is an isolated integration derivative of `tdleaf-human-prior-own-v2-qualified-producer-v3`. It does not edit that frozen producer, its two live collection registrations, or any collection output. It preserves the TDLeaf model, lambda-target conversion, prior+residual objective, Adam math, and native format.

## Integration repairs

- The frozen `prove.py` imports `module` from `convert.py`, although the converter exports `load_module`. This copy aliases `load_module` under the expected local name; the fresh-process proof uses the same hash-pinned loader implementation.
- Frozen `train.py` imports `pins_tree` from `convert.py`, but the converter only defined the singular `pinned`. This copy provides the recursive hash-verifying spelling expected by `train.py`.

The fixes are recorded as a new source closure. They do not rewrite the frozen v3 inventory or claim that its original native proof ran.

## ROOT-only actual sequence

Do not run any step until each collector closes with `status=PASS-exact-row-budget` and its immutable `receipt.json` exists. Use only the v4 registration/seal, actual receipt, events and alias chunks under the registered `/dev/shm/harbichess-human-prior-tdleaf-v2/g-1/<seed>` paths. Do not substitute v3 actor artifacts or teacher weights/labels.

1. Run `audit_collection_six.py` against the actual registration, receipt, events and clock for each seed. Require all six chronological root/PV packets and full alias/path protection checks.
2. Build a `human-prior-tdleaf-dataset-conversion-seal-v2` from those exact refs; `producer_directory` points to the original frozen v3 producer because the actual registration binds those source files. Use the registered protected aliases, alias chunks, and the registered parent model/prior helpers.
3. Run `convert_cli.py --seal <sealed-conversion.json> --output <new-RAM-directory>`. It performs deterministic, read-only replay and publishes dataset, provenance and a hash receipt in RAM.
4. Build `proof` and `fresh-fit` contracts from the same converted data/provenance and actual six-packet audit. The contract builder validates the actual H0 parent admission, not teacher256. The proof uses the original 600-second phase clock and checks 8 / 4 + fresh-process resume to 8 plus six strict native loads. Fresh64 requires that same seed's proof; ROOT should register/run both seeds' proofs before either fresh-fit.
5. Run the copied `prove.py` with a ROOT-authored immutable phase registration. Its proof/fresh-fit outputs stay in a new `/dev/shm` directory. Fresh64 uses the same H0 candidate as an explicit weights-only initializer, with new Adam/RNG state. The result is qualification, not a strength claim.
6. Only after both actual proof+fresh-fit chains pass should a separate H0-aware known160 adapter admit the models. It must validate both original TDLeaf contracts and parent admissions; no teacher-parent adapter is valid here.

No actual conversion, proof, fresh-fit, search, or arena was run from this derivative when it was frozen. The synthetic tests exercise the repaired Python interfaces only.
