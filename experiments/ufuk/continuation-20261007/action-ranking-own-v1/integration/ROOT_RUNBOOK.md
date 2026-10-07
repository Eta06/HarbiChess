# ROOT runbook (prospective only)

This prototype is sealed separately from afterstate-own-v1. Root must freeze
the source bytes, source commit, data bytes, same-seed parent candidate, and
new phase clocks before any execution. The clock values below are supplied by
ROOT as JSON fields; this helper never chooses or extends a clock.

## 1. Build the conversion seal

The request JSON passed to `seal_factory.py conversion` contains:

```json
{
  "afterstate_seal": "/path/to/completed-afterstate-conversion-seal.json",
  "afterstate_result": "/dev/shm/afterstate/result.json",
  "afterstate_converter": "/workspace/HarbiChess/experiments/ufuk/continuation-20261007/afterstate-own-v1/source/convert.py",
  "first": 0,
  "deadline": 0,
  "operator_end_epoch": 0,
  "seed": 0
}
```

The numeric zeros are placeholders in this example and must be replaced by
ROOT's observed, immutable values. The factory verifies the afterstate result
and the original ownQ source seal/registration and emits a publish-once
`NNUE-own-action-ranking1024-conversion-seal-v1`.

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python \
  integration/seal_factory.py conversion \
  --request /workspace/work/harbichess/continuation-20261007/action-ranking-own-v1/registrations/conversion-request.json \
  --output /dev/shm/harbichess/action-ranking/conversion-seal.json

PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python \
  source/convert.py \
  --seal /dev/shm/harbichess/action-ranking/conversion-seal.json \
  --output /dev/shm/harbichess/action-ranking/data
```

Conversion performs no inference, search, or SGD. It replays source histories,
checks every legal move, and validates each nonterminal child alias against
the corresponding original root search's pinned alias segment. Conversion
must fail if any such child is absent. It publishes data/provenance/result
under RAM only, with a 15 GiB cgroup cap, 256 MiB workspace floor, and the
single registered 600-second conversion window.

## 2. Build and prove the new typed phase

After ROOT verifies the conversion receipt and registers an immutable proof
clock, build request JSON must supply the data/provenance paths, same-seed
teacher candidate and teacher contract, prior helper, feature schema, clean
core repository/commit, and complete inference source SHA map. It also names
the exact proof first/deadline/operator-end values and `mode="proof"`.

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python \
  integration/seal_factory.py build --request proof-build-request.json \
  --output /dev/shm/harbichess/action-ranking/proof-build.json
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python \
  source/contract_builder.py --seal /dev/shm/harbichess/action-ranking/proof-build.json \
  --output /dev/shm/harbichess/action-ranking/proof-contract.json
```

Create the orchestration registration with the same explicit phase clock,
then run:

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python \
  integration/seal_factory.py orchestration --request proof-run-request.json \
  --output /dev/shm/harbichess/action-ranking/proof-registration.json
PYTHONDONTWRITEBYTECODE=1 /workspace/HarbiChess/.venv/bin/python \
  source/prove.py --registration /dev/shm/harbichess/action-ranking/proof-registration.json
```

The proof must pass whole 8 updates versus pause 4 plus fresh-process resume
to 8, with all six native payloads loaded in fresh processes. Only after
success may ROOT freeze a separate 1,800-second `fresh-fit` registration.
The fit starts from the same teacher weights, fresh Adam/global/private RNG,
and exactly 64 updates; it performs the same full 16-root objective. Proof and
fit clocks, dataset, parent, source closure, and optimizer state are all
version-bound. A failed phase is preserved and is never restarted under a
new clock.

`contract_builder.py` and `prove.py` both require full source/input SHA pins,
clean core Git state, free workspace ≥256 MiB, and the shared 15 GiB memory
guard. Training is CPU-only and writes checkpoints under `/dev/shm` only.

This prepares a testable method; it is not a strength result. It does not
replace the existing E8 baseline or change any frozen strength gate.
