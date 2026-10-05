# Independent full-critic native serialization review

Read-only review of the preserved V3 whole/pause artifacts and the corrected V4 proof. No source was changed and no fit/job was launched by this review.

## V3 diagnosis

The preserved V3 step-4 payload confirms the resume rejection was only one JSON representation mismatch:

- `whole/contract.json`: `contract["trainable_names"]` is a JSON list.
- `whole/checkpoints/step-00000004/checkpoint.json`: manifest contract also has a list.
- `training.pt`: `payload["contract"]["trainable_names"]` is a tuple, while the separate top-level `payload["trainable_names"]` is correctly a tuple for `restore_native`.
- Recursive comparison found `trainable_names` as the sole difference between the JSON contract and the torch payload contract. It contains 26 entries.

That explains the failure at the contract equality guard before native restore. Storing `list(trainable_names)` in the JSON-facing contract is the minimal repair. Canonical JSON hashes are unchanged because Python JSON serializes both tuple and list as the same array. The native payload's distinct top-level names should stay tuple-normalized.

An in-memory serialization regression passed: the corrected contract survives canonical JSON roundtrip, matches the checkpoint manifest contract, and still has names equal to the tuple-based native parameter-name field. The protocol trainable-name digest is unchanged.

## V4 read-only result check

The corrected V4 receipt reports `PASS-native-restart-not-fit`, 8 accepted updates, and all six strict fresh-process loads. It finished at epoch `1791226135.319565`, before the unchanged deadline `1791226458.5434084`. I independently byte-compared whole8 and split/resume8 `training.pt` payloads; they are identical with SHA-256 `b53d9f33c48101728e20a149bb8bbff72837af5dd61b067a014e2d18e88d180f`.

The production freeze check sorts parameter names in both hashes and compares exact frozen parameter bytes. The full native restore also checks contract, trainable-name tuple, trainable parameter keys/shapes/dtypes, frozen-parameter hash, optimizer step, and Python/NumPy/Torch/sampler RNG state. No additional tuple/list mismatch was visible in the V3 persisted contract/native payload.

This is a serialization/restart qualification only; it provides no fit-quality or strength evidence.
