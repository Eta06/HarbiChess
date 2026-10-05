# UFUK A100 reconnection and interrupted CPU state — 2026-10-05

User renewed permission after the original 09:00 Istanbul deadline. This receipt reports the historical attempt as interrupted; it does not retroactively extend its registered clocks or change strength thresholds. At this stage only connection/GPU and read-only artifact checks ran. No training, checkpoint update, runtime allocation or package installation occurred. The reason for the conversational usage interruption is not independently established by these artifacts.

SSH via existing `a100.emda.tr` Cloudflare endpoint succeeded. Presented ED25519 fingerprint exactly matched the user-confirmed `SHA256:TD0gsv0YY3Bt/F8GswmkBj3pAMQ2123Z68W+/8cWX/o`; a separate known-hosts file preserves the old host record. Host81c9b5623364/boot570719ad-af7f-43bd-adb3-2d014b1fce11 differs from the previous runtime. With the authorized LD_LIBRARY_PATH, nvidia-smi succeeded: A100-SXM4-40GB,40960MiB, driver580.82.07. PyTorch2.11.0+cu130/Python3.13.15 detected one CUDA device and a real16×16 matrix multiplication returned the expected16. No private-key contents were read, printed, changed or copied into this record.

The explicitly checked previous project paths were absent on the new host. This limited path check does not prove all old remote files are lost; restoring source and persistent artifacts is needed before production can be assumed ready. No HarbiChess training process was observed in the checked entrypoints locally or remotely. Progress files marked running are stale and do not establish a live process.

## Passed tests and remaining failures

- CPU tiny CLI qualification PASS273.79seconds: uninterrupted2epochs vs pause1/freshresume2;20checkpoint/journal byte matches and all6native fresh-process strict loads. [Original receipt](../../experiments/ufuk/cpu-contingency-v1/CLI-qualification-result/result.json).
- CPU fullshape E1 qualification PASS367.06seconds under original900:64actors×256steps,16384legal transitions, actual chronological/neural packet checks and six corruption rejections.12retained updates,2059own-search targets and10965retrospective completed-game WDL rows. [Independent original proof](../../experiments/ufuk/cpu-contingency-v1/formal-owned-launch/E1-qualified/independent-E1-audit.json).
- Earlier A100 source8 suite remains93PASS/2FAIL/0SKIP; the two missing-reference-fixture failures were never converted into95PASS.
- CPU formal seeds20261925/20261926 each closed native epochs0,1,2 and32768own-play transitions. Accepted updates24/21; only independent epoch0/1 audits completed. Fixedepoch8 was never produced. At04:09:52.958579UTC the cohort recorded INCOMPLETE after seed25 independent fresh CLI replay returned1: registered fullgame memory ceiling exceeded. Owner cleanup stopped the owned production/auditor groups. Preserve all original timers, receipts and partially interrupted state. This turn verified both closedepoch2 seven-file inventories and exact frozen-epoch/journal SHA equality; it did not claim strict-runtime replay or epoch2 full-audit completion.
- The CPU tiny full-state deduplicated archive was locally packed/hashed,49,824,432bytes/SHA0809d667d126692b3de431aa7de3e8c6be2737212c862b93aa32b9b8e5876f1c. Release uploads returned HTTP400 Bad Content-Length; new CPU public persistence remains pending. Previously verified128public assets remain a separate historical persistence proof.

## Strength conclusion

**No independently established real self-learning strength gain.** The last completed Main40 strength experiment used576heldout games and failed both frozen-seed gates. Direct scores against e8 were.52604/.51563,98.75%intervals[.41146,.64063]/[.41146,.61979]; paired Stockfish512 gain+.00521/−.01042 with intervals crossing zero. Final Stockfish scores.06771/.05729 also failed the absolute.25gate. [Unchanged failure report](UFUK-main40-selflearning-strength-result-20261005.md).

The strongest independently validated reference remains teacher-origin e8, portable checkpoint SHAe8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03, rehashed in this turn. Latest CPUepoch2 trained candidates are preserved but unqualified and unpromoted. Functional inference/training/checkpoint tests are infrastructure evidence, not proof of self-learning chess improvement. AppleMetal remains untested. No Stockfish superiority or original-paper contribution is established.

## Continuation

First restore verified source/artifact geometry to the renewed A100 and diagnose the CPU memory-guard/concurrency failure. Preserve full optimizer/RNG/replay state and distinguish exact full-resume from a weights-only bootstrap. Record a new prospective continuation/validation protocol; retain the old INCOMPLETE result, original strength thresholds and statistical family accounting. Complete native/data/resume and backend audits before independently evaluating fixed candidates. Do not reuse baseline outcomes to pick candidates or call a training-loss change success.

Machine-readable connection receipt, historical failure and both exactepoch2 inventories: [verified state](UFUK-A100-reconnection-state-20261005.json), SHA143a17576ed850c8d798641b380b02d31c9f0cc0899670d9337c85af561ad9cb. This is state recovery and connection verification, not a new training experiment.
