# Forensic own-learning native v4

This is a typed native/training continuation over the already-collected two-seed procedural own-search data. The original collector, source registrations, receipts, events, and six-packet search audits remain unchanged. The collector's receipt points at the wrong registration hash because its helper shadowed a path variable; v3's derived receipt view records both the raw pointer and the verified original registration. The view does not rewrite or relabel the receipt.

Both actual six-packet audits passed. The complete ROOT audit manifest is
`/workspace/work/harbichess/continuation-20261007/procedural-forensic-v4-actual-audit-set.json`.
The unchanged v3 converter then independently replayed the full history and produced:

| Seed | Dataset SHA-256 | Provenance SHA-256 |
| --- | --- | --- |
| 20262905 | `519a9cc9887751f5042ae159e41e5929e22e740d7e3df49c1be61daaf699e24c` | `3c37f0460efaee899a1d12e09b9f948523e8f656b389d51666b7327a68ab22f9` |
| 20262906 | `13ca3b5acd37b269529184b9eaaae7c9243612877d16662d2e03d48cd96c1567` | `bbf48d09d463b6b021230396e36414dccc2ec6d57b9514dd1c3b8f58575a552c` |

The dataset is 1,024 own-search rows per seed. It contains no new teacher labels. Caps and unfinished tails remain unknown; they are not draws. The exact raw sources and all aliases are carried by the converted provenance.

## Native identity and proof

The v4 native, training-contract, and execution-scope schemas are distinct from v1-v3. The numerical `Learner.advance` AST and `MATH` constants are tested against v1; only schema admission and forensic-data validation differ. Initializing from the admitted parent copies weights only and starts fresh Adam/RNG state. Resume is accepted only within the exact v4 contract.

Run `execute_forensic_native_v4.py` once per phase and seed through ROOT's owner/controller:

```sh
PYTHONPATH="$PWD" /workspace/HarbiChess/.venv/bin/python execute_forensic_native_v4.py \
  --root /workspace/work/harbichess/continuation-20261007 \
  --seed 20262905 --mode proof --cpu-core 1
```

The proof phase is bounded to 600 seconds and checks whole8 against pause4/resume8 plus six fresh-process native loads. Only after both seed proofs pass, run `--mode fresh-fit` for each seed. Fresh-fit is bounded to 1,800 seconds, makes 64 updates from the same dataset and parent, then opens fresh0/final64 native states. The runner preserves logs and failed receipts; it enforces the 15 GiB memory budget, 256 MiB disk floor, CPU affinity, and phase deadline. It writes each contract beside the phase output, so contract generation cannot pre-create the trainer's fresh output directory. It does not run strength games.

## Current status

The actual audits and full-history conversions passed. Read-only contracts and registrations were successfully built from both real dataset/provenance pairs. Synthetic native/adapter tests and helper import checks pass. The native proof and fresh-fit phases have not been run by this implementation stage, and no strength result is claimed.
