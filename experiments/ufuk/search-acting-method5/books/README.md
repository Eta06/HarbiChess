# Method5 unused strength-root books (roots only)

These are two prepared 48-root books for qualification-ledger slot 5, not a registered or launched training experiment. No model/engine calls, games, outcomes, SGD, or optimizer updates were used. Selection seed 20261605 maps to training seed 20261525; selection seed 20261606 maps to 20261526.

Both runs use the existing `history_openings.freeze` implementation from clean consumer commit `635413e426ef5b09cb76073c0955898912247466` and the same original PGN SHA-256 `5b8c29f91628ed27d468dc6355bf3eaad98a03bcdf638c32f3c3106bff162c8a`. Each selection had a 600-second ceiling and independent replay was subject to the same 900-second whole ceiling. Measured wall time was 48.63 seconds for seed 20261605 and 48.46 seconds for seed 20261606.

Exclusions include capacity source/root/full-prefix receipts, all current historical `docs/research/*opening-splits*.json`, the formal2 train book and both formal2 strength books, both confirmation3 books, previous strength-second book, both method4 books, and both method4 inherited source-ID and full-prefix position-key lists. Seed 20261606 also excludes every method5-20261605 source ID and all of its root-through-prefix position keys. Root source inputs, existing method4 books, prior exclusions, current docs books and prior teacher/self-play book paths are SHA-recorded in each audit.

Each selected record was independently reparsed from the original PGN and checked for record hash and source ID, standard start position, both ratings >= 2000, base time control >= 60, root ply in `{16,24,32,48,64,96}`, legal complete UCI prefix, exact full FEN, valid position, and nonterminal/undrawn root. Both books contain 48 records; source overlap and root-position overlap are zero. The cross-audit confirms all 48 seed-20261605 source IDs and all 48 root keys appear in the seed-20261606 exclusion lists, as do all 96 method4 source IDs and 2,865 method4 full-prefix keys.

| Selection seed | Training seed | Book SHA-256 | Audit SHA-256 | Selection seconds | Total seconds |
| --- | ---: | --- | --- | ---: | ---: |
| 20261605 | 20261525 | `ac6cdba2238d376e22e3db2ee1ba63773892bf6493b66360f76a9b6df8ab8117` | `d7c90e7537ed6543c8306f18557ea1d12d22bbb31e29067e1b19ddcc52827042` | 29.19 | 48.63 |
| 20261606 | 20261526 | `de740d4977a4938d1769cb861682a53f86ee39201ab79a2b13b9a2095b1bf5e4` | `a91a36b4c191db487e65ea7c44635042b4edc92fcaae0563f4c3a5a9922b36d9` | 28.75 | 48.46 |

Primary receipts are `20261605/audit.json`, `20261606/audit.json`, and `combined-audit.json`. Books and full exclusions are stored alongside each receipt. `freeze-method5-book.py` freezes one seed at a time; the second run refuses to start until the first book exists. `cross-audit-method5-books.py` independently checks cross-seed and method4 exclusion coverage.

Reproduction used `/workspace/HarbiChess/.venv/bin/python` with `PYTHONPATH=/workspace/work/harbichess/search-acting-method5-books/consumer-635413/src`. This directory and its consumer clone are scratch only; no source or prior evidence files were changed.
