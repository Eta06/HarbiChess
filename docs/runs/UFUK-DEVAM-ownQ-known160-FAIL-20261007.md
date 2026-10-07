# UFUK-DEVAM: ownQ v2 fails the development strength screen

All160 games closed at epoch1791368759.0008585, within the original7200-second phase started1791366266.8496559. A metadata-only recovery added the missing original E8 value-helper pin before any search/game outcome. Original failed wiring, clocks and source bytes were retained; recovery did not retrain or change gates/search.

| Seed | Own64–E8 | Own64–exact parent | Own64–SF512 | Parent–SF512 | E8–SF512 |
|---|---:|---:|---:|---:|---:|
|20262905|.6875|.40625|.03125|0|0|
|20262906|.34375|.5625|.0625|.03125|0|

Each cell is16 games, the same8 known opening roots paired in both colors. Both seeds FAIL the unchanged screen. No cap occurred. Small Stockfish draw-score differences do not constitute successful self-learning. In particular both fail SF>=.25 and direct-parent>.60; seed20262906 also fails direct-E8>.60. No independent confirmation was drawn and no strongest-model/promotion claim is made.

ROOT independent audit replayed all13,072 continuation plies, legality, complete history, terminal winner/draw and color/node accounting. Status PASS-fullhistory-integrity; strength screen FAIL. Stockfish nominal512 uses Threads1/Hash16/ClearHash. Actual Stockfish total1,658,661 nodes and HarbiChess5,038,835 search nodes are logged; nominal UCI node limits are not represented as strict actual equality. Per-move overruns remain recorded. Six fixed chronological search packets (two seeds × own64/parent/E8) were actually recomputed and matched both profile and arena move/value-HEX/nodes/evaluations/depth/root actions.

Root audit first1791368930.4177356, end1791369830.4177356, actualfinish1791368935.9169867. The corrected audit enforces played depth1..8. This is the known-book selection screen, not population-independent strength evidence. Eight opening families and the descriptive paired intervals do not support a publication strength claim.

Integrity receipts, clocks and exact source/command hashes are in `experiments/ufuk/continuation-20261007/actual/known160-root-independent-audit`. The source/actual profile contract is `nnue-strength-v2-e8-wiring-recovery/protocol.json` (SHA22f87ce706483bcc1364a650a405b7edbf9b2fdccd380741a24a12fd2a29ddb2). Models: own64 SHAs287df0c1ff5779e0ee4cc9ab9df38b077c71daee298dff376f2d819e4a2f27a4 and8627e94ca0cd6a177f3644b57ce89f30873b03b8cc99363fdf7663dddae05328; teacher-parent SHAs52e029... and4ff008...; E8 referencee8fe6d4... . Frozen core commit6fcc8b476d25495d1c9c413e55b2c7ba4794013e; original searchde53c147... . Exact full hashes remain in the machine-readable pins.

Diagnosis: own-Q imitation is not demonstrably improving play, most of its rows are UNKNOWN, and the teacher-trained baseline itself is weak. The next preregistered tests change target propagation and completed-game reward coverage, while a separate search-only baseline probe tests move-ordering/search limitations. Neither hypothesis is assumed successful.

Checkpoint preservation: V10b's19 public Release parts were independently fetched and reconstructed. CapsuleSHA bbb42d35ca972140331b15b66a8e9ea74aae3316bb4bf99c6dd3bfd353ad3844; all261 original files200,118,263 raw bytes passed size/SHA verification. This snapshot includes own collection/native optimizer/RNG/model/proof/fit/source state before arena closure; it is byte preservation, not a cross-runtime expired-clock resume claim. OldV9/V8/V7/V6 and failedV10 remain preserved. Apple hardware was not available; MLX production code was not silently replaced and remains untested by this CPU experiment.
