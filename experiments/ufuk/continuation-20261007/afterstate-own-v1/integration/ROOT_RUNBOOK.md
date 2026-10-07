# ROOT afterstate ownQ-v2 phase wiring

This is an unexecuted recipe. It does not pick clocks or start work. ROOT must
register new conversion/proof/fit windows under the existing operator end
`1791448916.685839`; an expired source ownQ window is provenance only and is
never reused. Each output path is new and publish-once under `/dev/shm`.

The proposal is a same-parent, same-1,024-row target transform. It adds no
self-play, search, or teacher calls. The original ownQ-v2 source converter
remains authoritative and is replayed before any afterstate bytes are emitted.

## Conversion

The source ownQ-v2 spec/result are already-finished inputs. ROOT supplies an
observed first/deadline pair for a new 600-second conversion phase:

```sh
BASE=/workspace/work/harbichess/continuation-20261007/afterstate-own-v1
PY=/workspace/HarbiChess/.venv/bin/python
SEED=<20262905-or-20262906>

$PY "$BASE/integration/seal_factory.py" conversion \
  --source-spec /dev/shm/harbichess-NNUE-own-data-v2/$SEED/conversion-seal.json \
  --source-result /dev/shm/harbichess-NNUE-own-data-v2/$SEED/result.json \
  --source-converter /workspace/HarbiChess/experiments/ufuk/continuation-20261007/nnue-own-training-v2/convert.py \
  --core-repo /workspace/HarbiChess \
  --first <ROOT-registered-first> --deadline <ROOT-registered-deadline> \
  --operator-end-epoch 1791448916.685839 \
  --output /workspace/work/harbichess/continuation-20261007/afterstate-own-v1/$SEED/conversion-seal.json

$PY "$BASE/source/convert.py" \
  --seal /workspace/work/harbichess/continuation-20261007/afterstate-own-v1/$SEED/conversion-seal.json \
  --output /dev/shm/harbichess-NNUE-afterstate-data-v1/$SEED
```

The converter returns only after the old converter replays to byte-identical
ownQ data/provenance and after the new source event stream has been replayed in
full. Unplayed protected rows stay in raw provenance but are never afterstate
targets. The result records the full source and afterstate histories, both
mover perspectives, terminal flag, source censor status, and target sign.

## Proof and fit

For each seed, separately register proof and fit clocks. The proof phase is
600 seconds and the fit phase is 1,800 seconds. Fit registration must be made
only after proof passes, while retaining the operator end and the fixed 64
updates. The seal factory checks that phase clocks exactly match their contract.

For each phase, build a seal with the `build` subcommand, then build the
contract and orchestration registration:

```sh
$PY "$BASE/integration/seal_factory.py" build \
  --mode proof --seed "$SEED" \
  --first <ROOT-proof-first> --deadline <ROOT-proof-deadline> \
  --operator-end-epoch 1791448916.685839 \
  --dataset /dev/shm/harbichess-NNUE-afterstate-data-v1/$SEED/dataset.json \
  --provenance /dev/shm/harbichess-NNUE-afterstate-data-v1/$SEED/provenance.json \
  --parent-candidate <same-seed-teacher-candidate.pt> \
  --parent-contract <same-seed-teacher-contract.json> \
  --feature-schema <same-seed-teacher-feature-schema> \
  --prior-helper <same-seed-frozen-prior.py> \
  --inference-source-json <same-seed-frozen-inference-source-map.json> \
  --core-source-repo /workspace/HarbiChess \
  --core-source-commit <frozen-core-commit> \
  --output /workspace/work/harbichess/continuation-20261007/afterstate-own-v1/$SEED/proof-seal.json

$PY "$BASE/source/contract_builder.py" --seal <proof-seal.json> --output <proof-contract.json>
$PY "$BASE/integration/seal_factory.py" orchestration \
  --mode proof --seed "$SEED" --cpu-core <ROOT-assigned-core> \
  --first <ROOT-proof-first> --deadline <ROOT-proof-deadline> \
  --operator-end-epoch 1791448916.685839 \
  --contract <proof-contract.json> --dataset <dataset.json> \
  --target-provenance <provenance.json> --parent-candidate <teacher-candidate.pt> \
  --build-seal <proof-seal.json> --source-dir "$BASE/source" \
  --output-dir /dev/shm/harbichess-NNUE-afterstate-proof-v1/$SEED \
  --output <proof-registration.json>
$PY "$BASE/source/prove.py" --registration <proof-registration.json>
```

The proof checks whole 8 updates against pause 4 plus fresh-process resume to
8, and strict fresh-process loads for all six payloads. Only after its result
passes may ROOT repeat the build/orchestration steps with `--mode fresh-fit`,
the separately observed 1,800-second fit clock, `--proof-result` and
`--proof-contract`. Run the same `prove.py` orchestration, which performs the
fixed 64-step fit and fresh strict loads at 0 and 64. The candidate and native
are admitted only by `arena_adapter.admit_afterstate_child` and the separate
afterstate audit record. Neither proof nor fit is a strength result.

## Typed strength record

After all conversion, audit, proof, and fit receipts exist, call
`seal_factory.make_child_record(...)`. `collection_receipt`,
`collection_registration`, and `events` refer to the original ownQ-v2
collection. Its `variant_helpers` map must contain exactly `model`, `native`,
`contract`, `convert`, and `contract_builder`. The shared protocol variant is
`afterstate-search-q-v1`; its independent audit set must use the original
ownQ-v2 twelve-search packet audit plus the new afterstate native receipts.
The adapter revalidates the target contract and refuses other phase/native
schemas. No result may be chosen by loss or arena outcome.
