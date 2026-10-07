# Selected-action afterstate own-search value proposal

This isolated proposal reinterprets the already sealed ownQ-v2 1,024-row
training set. It does not collect games or request new labels. For each row,
it replays the complete source history and requires that the logged best move
was actually played. If the child position is nonterminal, the target is the
negative of the clipped root search value because the value viewpoint changes
to the child side to move. If the child is terminal, the target is exact WDL
from that child viewpoint. A protected/discarded action is retained only in
source replay and can never become a training row. Unknown game endings remain
explicitly counted; own search Q is still a target for those rows.

The learner starts from the same-seed named teacher candidate weights, creates
fresh Adam/global/private sampler state, and performs a fixed 64 updates. The
native schema, phase, dataset, target rule, contract, proof, and orchestration
are versioned separately from ownQ-v2, MC, and TD(lambda). This is an
afterstate target representation ablation, not a novelty claim or a strength
result. The target transform is standard value-viewpoint bookkeeping; any
strength benefit must be shown by the original registered arena gate.

## Source layout and APIs

- `source/convert.py`: `convert(seal, guard)` replays the pinned original
  ownQ-v2 converter byte-for-byte and the raw chronological event stream.
- `source/contract.py`: `admit_contract(contract_bytes, data_bytes,
  provenance_bytes)` checks all 1,024 row IDs, censor counts, action-played
  flags, history digests, target signs, and phase/native version.
- `source/contract_builder.py`: `make_contract(seal)` binds proof/fresh-fit
  clocks, the exact converter input spec, dataset/provenance, parent, and code
  closure.
- `source/native.py`: `Learner`, `load_native`, and `bits_equal` implement the
  fresh-Adam full-resume state.
- `source/prove.py`: `execute(registration)` runs the fixed proof or fit phase.
- `source/arena_adapter.py`: `admit_afterstate_child(record)` admits a fully
  qualified child; `make_afterstate_value(record, protocol)` exposes its value
  endpoint through the existing arena.

## Root registration requirements

Conversion seal schema: `NNUE-own-afterstate1024-conversion-seal-v1`. It must
pin the original ownQ-v2 conversion seal, result, dataset, provenance,
registration, receipt, events, alias chunks, source converter, same-seed
teacher candidate, feature helper, prior helper, and the original core pin.
The seal is a new immutable clock registration; it must not reuse an expired
phase clock. Outputs go under `/dev/shm` and are publish-once.

Proof and fit use separate `NNUE-own-afterstate-training-orchestration-v1`
registrations and separate contracts (`proof` 600 seconds, `fresh-fit` 1,800
seconds). The proof must show whole 8 steps equals pause 4 plus fresh-process
resume to 8, and six fresh-process native loads. Only after proof passes may
the fixed 64-step fit begin, with new zero-step/final native loads. Neither
proof nor fit is strength evidence. No actual source rows, candidate weights,
inference, or training run were used while preparing this proposal.

Synthetic checks:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python -m pytest -q -p no:cacheprovider tests
```
