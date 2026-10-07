This is a separate SEARCH-ONLY development experiment, not self-learning or a virgin confirmation. It preserves original de53 and advanced v1 bytes. The prospective executor binds advanced **search_v2.py**, whose sole search change caches immutable root ancestry once and hashes exact within-root suffixes in the reset-only TT. Public/outside-search keys retain full-history semantics. The TT never survives a root search; full current FEN, chess960, phase and root-relative ply remain in each key.

ROOT must supply clocks and launch through its owned process-group catalog on CPU2. `register.py` performs metadata validation only; `develop.py` performs actual searches/games. No timestamps below are invented. Whole phase limits are profile600s and arena7200s, each capped by operator1791448916.685839. The executor uses both original wall deadline and its monotonic equivalent, checks aggregate inactive-file-v1 memory15GiB, disk floor256MiB, and RAM output256MiB. ROOT's external owner must enforce the same absolute whole deadline and terminate only verified owned descendants (including UCI engine). Failures are immutable result records, never a new clock.

Run metadata/import help first, then ROOT supplies these flags:

```sh
PY=/workspace/HarbiChess/.venv/bin/python
BASE=/workspace/work/harbichess/continuation-20261007/pv-history-search-v1
$PY "$BASE/qualification/register.py" --mode profile \
  --first ROOT_FIRST --deadline ROOT_FIRST_PLUS_600 --operator-end-epoch 1791448916.685839 \
  --core-repo /workspace/work/harbichess/cpu-additive-source-6fcc8b4 \
  --core-commit 6fcc8b476d25495d1c9c413e55b2c7ba4794013e \
  --old-search /workspace/HarbiChess/experiments/ufuk/cpu-budget-search-v1/search.py \
  --advanced-search "$BASE/search_v2.py" \
  --prior-helper /workspace/HarbiChess/experiments/ufuk/cpu-classical-own-v1/value.py \
  --train-roots ROOT_FROZEN_4096_TRAIN_POOL --incheck-extensions 1 \
  --output /dev/shm/harbichess-pv-human-profile-v2 \
  --registration ROOT_NEW_PROFILE_REGISTRATION
# ROOT owned launch command: $PY "$BASE/qualification/develop.py" --registration ROOT_NEW_PROFILE_REGISTRATION
```

Pool endpoints are the **first24 original rows**, no outcome/score selection or terminal filtering. All24 must replay legally to valid nonterminal states; otherwise fail. Pair order alternates to control warmup. All48 actual calls use unchanged human18 prior,512 charged nodes/q2/max8. Every ordered evaluator input archives exact full-rule-history SHA, conservative mirrored placement alias and float64 value in48-byte regular records. Full pre-action FEN/history and every result/counter are in progress/result; segment references and chunk hashes permit reconstruction. Timing includes actual trace construction; its measured cost is separately logged. It is not uninstrumented throughput.

Default profile must have median advanced/original wall ratio≤1.10 before arena admission. Arena registration uses the same flags/source references except `--mode arena`, `--deadline ROOT_FIRST_PLUS_7200`, `--book ROOT_EXACT_EXISTING_KNOWN8_BOOK`, `--stockfish ROOT_OFFICIAL_SF19_BINARY`, and `--profile-result /dev/shm/harbichess-pv-human-profile-v2/result.json`; omit train-roots. ROOT seals actual book/SF/registration SHAs before launch. Result contains exactly32 games:8 fixed openings×both candidate colors×two unchanged human-prior search implementations. Threads1/Hash16/ClearHash per game/SF nominal512; every actual UCI node overrun is logged. Cap400 is total board ply including opening, matching original. Caps remain UNKNOWN, with draw-filled and pessimistic summaries separately named. Known rules use full move stacks and claim_draw=True. All inputs/source files are rehashed at closure. This development score cannot satisfy final strength gates or establish learning.

Optional horizon0 is a **new separately ROOT-clocked profile-only** registration with `--incheck-extensions 0`; it cannot admit games. Default1 adds a bounded charged in-check evasion and reports static horizon returns explicitly. Advanced static all-root coverage reports completed_depth0; original de53 reports1 for its fallback. Neither is relabeled as a completed deeper iteration.

Future teacher-free initialization (proposal, no implementation/promotion here): create a NEW typed `human-prior-zero-residual-init-v1` phase and native0, using the same fixed NNUE architecture/init seed with literal zero output head. Verify every nonterminal value HEX equals unchanged human18 prior and all terminal values remain rules-owned. Store initialized embedding/head, empty newAdam, fresh global/private RNG, architecture/source/input hashes and named human-prior baseline; do not reuse teacher-bootstrap phase or relabel own64 as initialization. Own collection targets must come from that current zero-residual parent using SAME admitted advanced search, with no SF target reads. A subsequent model-weights-only parent bridge must explicitly initialize newAdam/RNG; fullnative resume is only within its own versioned phase. Require fresh whole/split/native proof, trained latency and BOTH seed fixed-search learned-vs-exact-zero-parent strength gates. PVS/TT/history ordering and own-search distillation are established methods, not novelty claims.
