# AYNA: whole-game teacher bootstrap and independent strength test

Registered before teacher generation or training. This is a new learning experiment,
not a reinterpretation of the failed PORT self-play run or a promotion.

## Evidence and hypothesis

The completed AYNA v2 development probe found STM value MAE 0.907 against budgeted
native Stockfish WDL, top-16 teacher move coverage 53/96, and a conditional
oracle-value regret reduction of 0.0812. Its eight reused development families are
not an independent strength test. Existing YAPI/OLCEK origin-only adapters failed;
simply replacing the dense head with a 1x1 head is not a new solution.

Hypothesis: broader strong policy/soft-WDL supervision can improve both heads of
the existing portable decoupled MIHVER model, without an engine at inference.
Control: PORT generation-000000, SHA a155be94c96227815a2cac6f235c06622a5fb1df93a215b4ba391be500aaaa61.
Initial teacher-assisted games are labelled bootstrap data, not learned self-play.

## Frozen data and target semantics

- Seed 20261003. `docs/research/AYNA-opening-splits-20261003.json`, SHA
  019c5b4a49ac287ca03effb65b33ed97daa2e702ddd4a6972491969e27912b24.
- Source: lichess-org/chess-openings, CC0; download receipts are in the split file.
  First-four-ply groups were merged if any eight-ply line transposes, excluding
  MERCEK development families. Freeze 48 training, 16 validation and 32 unseen
  arena families. No teacher quality/outcome was used to choose the lines.
- Two trajectories per train/validation family, one engine/engine, one frozen
  neural/engine (neural colour alternates by family). Four **spawned** workers,
  each one Torch and one Stockfish thread, engine Hash 16 MiB. All within the
  existing four-CPU/16-GiB allocation; no GPU or purchased compute.
- Maximum 160 plies, labelled every second ply with parity alternating by game.
  At labelled positions: Stockfish 19, 32768 requested nodes, MultiPV min(4,legal).
  Record actual nodes, last complete unbounded common-depth iteration, full root
  FEN/history, legal UCI/actions and native root-STM W/D/L.
- Soft policy over the teacher's four moves: softmax of native centipawn/mate
  scores at temperature 100 cp, mate score clipped to +/-10000. Soft WDL from
  the first PV is an engine reference target, never an observed terminal label.
- Unlabelled engine moves use 4096 nodes. Engine actors sample the teacher soft
  policy with probability 0.20 before ply 40; otherwise choose first PV.
  Frozen neural actor uses its legal raw policy. No teacher assistance at arena.
- Version 1 oracle JSON records are separate from legacy replay schemas. Capped
  game outcome remains unknown. Whole families and whole games remain split;
  duplicate current-position keys between train/validation are removed from
  the validation learning panel and counts retained. Arena positions remain frozen.
- Dataset wall ceiling 1800 seconds; immutable completed per-game files permit
  restart after interruption. Stop on illegal moves, malformed/unbounded targets,
  missing provenance or nonfinite values. Preserve partial/failed runs.

## Learning and resume

- Unchanged decoupled architecture, all parameters trainable; AdamW LR 2e-4,
  weight decay 1e-4, gradient clip 5, policy/WDL weights 1/1. Batch 64, Torch CPU
  one thread, seed 20261003. At most 6000 updates or 1800 seconds per invocation.
- Evaluate all held-out rows every 200 updates (and initial). Select minimum
  validation policy-CE + soft-WDL-CE, with improvement threshold 1e-4. Early stop
  after 1000 updates without improvement; stop immediately on nonfinite update.
- At step 200, exit and resume in a fresh process. Full checkpoint must restore
  optimizer, Torch sampling RNG, cursor, configuration, verified dataset hashes
  and best-selection state. Warm-start weights alone are never called resume.
- Qualification for a subsequent arena: validation policy CE and value CE each
  improve by at least 0.10 vs initial; report teacher-Q MAE and top-16 coverage.
  Qualification is permission to measure strength, not proof of strength.

## Independent strength, speed and subsequent self-learning

- Frozen 32 arena families, both colours (64 games), max 240 plies, seed 20261003.
  Best qualified model vs initial, and each vs Stockfish 19 (one thread, 16 MiB,
  32 nodes). Neural search 16 simulations, deterministic Gumbel scale 0.
- Record every legal move, cap/outcome, actual wall time and evaluations. Report
  paired-family bootstrap and conservative interval, no general Elo inference.
  Fixed neural simulations and engine nodes are unequal-cost diagnostic budgets.
- Evidence of improvement requires candidate score >0.60 against initial and
  paired bootstrap lower bound >0.50, no >10% capped games; vs Stockfish scores
  must be reported even if worse. Do not use arena results to select a checkpoint.
- After qualification, preregister a separate bounded true self-play phase from
  the selected weights, explicit optimizer reset and held-out complete games.
  Require independent game-strength gains before claiming self-learning success.
- A failed qualification or failed arena remains failed. Follow-up changes require
  new registration with new unseen evaluation families; do not rerun until lucky.

Checkpoint/data hashes, source commits, failures, compute, restoration tests and
Release upload/readback status must accompany the result. Apple Metal/CUDA remain
untested on this CPU host; old MLX/replay/checkpoints are preserved.
