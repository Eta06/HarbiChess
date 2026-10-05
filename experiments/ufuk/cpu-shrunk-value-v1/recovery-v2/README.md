# SHRINK arena recovery v2 — scratch proposal, no jobs launched

The original helper, protocol, arena directory and all logs remain immutable. `resume_run.py` creates a separate recovery root and retains the original owner contract, original whole first/deadline and original qualification. All complete group JSON files and owner receipts are referenced in place by exact SHA. Future groups receive their actual launch first and `min(original_whole_end, first + original_per_group_seconds)`.

The interrupted group's invocation and progress are SHA-bound. Its **original** first and deadline are reused; no recovery clock reset. Completed game_end objects are copied exactly, and the original raw game_end line is retained in import events. Only an NN-only partial game may continue. The prefix is chronologically re-searched from the original complete opening history, at the original fixed node/search settings, and every selected move must match before continuing. Hidden Stockfish TT is not restorable: SF partials fail closed.

Old progress contains moves but no partial search packets or latencies. Reconstructed packets are explicitly labeled `actual-recovery-prefix-research`; their node/evaluation cost is separately reported as duplicate NN work. Old partial latency is UNKNOWN. All recovered game wall values measure actual recovery execution only; original completed games retain their original wall values. No elapsed interruption time or lost previous NN work is pretended to be zero or included in a fabricated measurement. The group header retains original first/deadline and states recovery start separately.

`audit_v2.py` preserves the original 160-game set, full-history rules/counters and numeric development gates. It validates the actual recovery executor SHA against a new explicit contract field, and the original parent helper SHA independently against original inputs; it never pretends the new executor has old bytes. Complete groups retain the original schema/executor binding. Recovered partial packet sequence, original complete game objects, duplicate work and input/clock bindings are additionally checked. Descriptive latency aggregates mix original completed packets and explicitly identified recovery measurements; they are not a new unchanged-latency claim. No formal strength claim is added.

Root reviewed launch shape (DO NOT EXECUTE automatically):

```sh
PYTHONPATH=CLEAN_SOURCE/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHON resume_run.py --checkout CLEAN_SOURCE \
  --study /workspace/HarbiChess/experiments/ufuk/cpu-shrunk-value-v1/arena \
  --qualification ORIGINAL_QUALIFICATION.json --output NEW_RECOVERY_ROOT \
  --first-epoch 1791217012.4869661 \
  --parent-arena /workspace/work/harbichess/cpu-shrunk-value-v1-actual/arena \
  --parent-contract-sha256 ORIGINAL_OWNER_CONTRACT_SHA \
  --recovery-helper-dir THIS_FROZEN_HELPER_DIRECTORY
```

The copies `search.py` and `value.py` must retain exact original hashes. Root must bind all supplemental helpers into a versioned recovery registration before launch and retain the real guardian interruption receipt. Root's genuine owned wholedeadline/process/disk guardian remains required. No new training or model selection occurs. `build.py` is a historical generation aid and **must not be deployed/run**: reviewed runtime files are independently frozen after fixes.

Tests use tiny fake-search fixtures, not real NN qualification. They check exact closed game import, strict NN-prefix correspondence, SF partial rejection, original clock/no restart, actual recovery main CLI wiring, duplicate counters and honest new-vs-old executor SHA.
