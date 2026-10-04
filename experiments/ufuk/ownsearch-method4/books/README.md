# Method4 virgin root books (roots only)

Prepared prospectively for qualification-ledger slot 4. These are two unused 48-root books, not a registered experiment. No inference, engine query, game, optimizer update, or outcome inspection was performed. The exact training-seed mapping is selection seed 20261505 → training seed 20261425 and selection seed 20261506 → training seed 20261426.

The freeze reused `harbichess.training.history_openings.freeze` and the original bounded PGN source. Before selecting the first book, exclusions included the capacity source/root/full-prefix sets, the formal2 train/validation book, formal2 48-root strength books for both seeds, confirmation3 books for both seeds, the previous strength-second book, and every opening-splits book under `docs/research`. Seed 20261506 additionally excludes every source ID and every root-through-prefix position from seed 20261505. Each audit records a SHA-256 inventory for its inputs and exclusions.

Each of the 96 selected records was independently reparsed from the original PGN and checked against its record hash, Lichess source ID, standard initial position, both ratings >=2000, base time control >=60, selected root ply in {16,24,32,48,64,96}, exact complete UCI prefix/FEN, legal moves, position validity, and nonterminal/undrawn root. Source headers, full move prefixes, full FENs and record hashes remain in the frozen books/audits. Both 48-root sets have 0 source overlap and 0 root overlap; all 48 first-book source IDs and all 1,416 first-book prefix positions appear in the second seed's exclusion lists.

The unchanged per-seed 600-second selection / 900-second total selection-plus-independent-audit ceilings were not approached: seed 20261505 used 29.14 seconds for selection and 46.85 seconds total; seed 20261506 used 28.43 seconds and 46.24 seconds. The same clean consumer repo commit `635413e426ef5b09cb76073c0955898912247466` and identical freezer/source hashes were recorded for both. This does not pin or authorize a future method4 source/runtime; those must be frozen separately before any launch.

Primary receipts are `20261505/audit.json`, `20261506/audit.json`, and `combined-audit.json`. Books are `20261505/opening-splits.json` and `20261506/opening-splits.json`. Frozen book SHA-256 values:

- 20261505: `3fcb0d8e7542f93a093de4387a0cb15eb2862dcfc72291cab22832c4847e16ca`
- 20261506: `39f073aa0f8a79431550b3295ab292518a8f0229c16910c474166340af5b3cff`

Combined cross-audit SHA-256: `bf2cf366ebbf1b3914894f85c4ff63b7a07ecceea8d2152a17cfddf4b8c5ba15`.
