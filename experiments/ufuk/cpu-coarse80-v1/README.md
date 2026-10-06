# Coarse 80-parameter own-Q residual (PREPARATION ONLY)

This is an isolated proposal. It has not loaded production labels, trained on chess data, evaluated a board with the new weights, searched, played games, or passed the project's runtime/strength gates. The parent agent must freeze source/helper/input hashes, profile latency, and run the real native qualification before any fit or arena.

## Fixed model and signal

- Frozen base: the existing 18-feature `ClassicalValue` prior, unchanged.
- Residual: 5 non-king piece tables × 4 × 4 square zones = 80 scalar parameters. Piece squares use each color's forward orientation; signs are from the mover's perspective.
- Each piece table is projected to zero mean. This removes a constant piece-type offset; it does not guarantee every board's material contribution is invariant.
- Initialization is exactly zero. The evaluator's zero branch delegates to the existing prior method, preserving the prior's output bits and avoiding extra feature work there.
- Training signal: only frozen 8,192-node own-Q targets, no Stockfish labels, no MC return term in v1. Capped, truncated, or unknown experiences cannot be added as Q targets. A later terminal-return term would be a separately preregistered ablation, never a fallback inserted after seeing results.
- Fixed recipe: 64 Adam updates, batch 256, lr .001, beta .9/.999, epsilon 1e-8, clip 5; game-uniform then row-uniform sampling; anchor .1, ridge .0001, 4-neighbor smoothness .0001. No validation-driven checkpoint selection.
- Inference C kernel uses strict IEEE compilation (`-fno-fast-math -ffp-contract=off`) for residual calculation. The existing Python prior feature extraction and sum order remain authoritative. The compiled path has only a synthetic bitboard parity test; the evaluator must still pass a real paired latency gate against both fast prior and E8 before launch.

## Provenance and native state

`registration-template.json` requires a clean core commit, exact Q label artifact SHA/schema/row count, prior helper/model hashes, own-search producer hash and 8192/q2/depth8 settings, source-closure file hashes, separate 600-second proof and 1800-second fit clocks, and a dedicated RAM output path. It is intentionally DRAFT; do not validate it as registered.

`learner80.py` stores all 80 weights, Adam m/v, step, immutable contract, candidate schema, sampler RNG and Python global RNG. Resume rejects a changed contract, parameter shape, candidate, missing state, or noncanonical native bytes. No old checkpoint migration is implemented: an old model may initialize a new experiment only under a separately frozen weights-only start with fresh optimizer/RNG, never by claiming full resume.

## Local checks

- `test_coarse80.py`: synthetic feature orientation, zero-mean constraint, exact zero passthrough, C-kernel parity, native corruption checks.
- `test_dataset80.py`: a synthetic full-history Q packet conversion fixture; it is not production provenance.
- `proof_synthetic.py`: whole 8 updates versus pause at 4 and a fresh subprocess resume to 8, exact native byte equality, then six strict native loads. All data are generated in the script.
- `registered_cli.py --help` checks importable CLI wiring. `--synthetic-native-proof` is non-production only.

A real run still needs an independently verified actual-label converter execution, pinned registration, source/helper closure, output/resource/deadline guards, actual full native proof and fresh-process loads, and unchanged two-seed 512-node strength gates. No strength or novelty claim is supported by this prototype. Closest prior art includes classical piece-square evaluation, TDLeaf-style search/return combinations, Expert Iteration, AlphaZero, and NNUE; this is not asserted as novel.
