# NNUE own-search collector v2 (source prepared, not run)

This is a ROOT-registered CPU producer. It records experience from the same-seed 256-step teacher-bootstrap NNUE candidate playing both colors. It does not read teacher scores while collecting, call Stockfish, train, or claim self-learning/strength before a later learner and arena gate.

## Frozen collection design

For seed 20262905 or 20262906, `metadata_factory.py` projects the actual final 4096-row TRAIN selection from `/dev/shm/harbichess-NNUE-teacher8192-v1/{seed}/selection.json`, binding the corresponding `labels-00004096.json.gz` SHA for provenance only. Each row preserves its source row/trajectory ID, root FEN, and full UCI history; the factory replays every history and checks its FEN and digest. It also binds the frozen Oct6 teacher profile/admission, same-seed candidate identity, exact compiled NNUE/prior/search/core source hashes, and ROOT supplied protected-alias inventory. No learner reads the teacher-label file.

The collector's deterministic selection salt/order is unchanged from v1. It selects 128 outcome-blind starts from the 4096 TRAIN roots, simulates both sides with one frozen parent and runs exactly 8192 nodes, qdepth 2, max depth 8, at each nonterminal action. It stops after exactly 1024 usable rows, within at most 128 starts and 16 plies per start; if protected-state exclusion or natural endings leave fewer rows, it fails. It never substitutes validation roots.

A row contains root FEN/prefix, current full history before the selected move, mover, raw mover-perspective search Q, clipped Q, selected move, node/evaluation/depth/root-action counts, and every actual static-NNUE input alias. `mate_range_score_returned` only means the returned search score has absolute value above 1; it is not an independent mate proof. Selected search moves are actually played unless a protected evaluator input is detected (`selected_action_played=false`). Any game whose path board or evaluator input reaches a protected alias is excluded in full from learning. Final path boards are recorded and checked even at ply16, terminal, or row-budget stop. Completed self-play games attach mover-perspective WDL from their own terminal. Ply-cap and row-budget tails remain UNKNOWN, while their verified own-Q rows remain usable. No action-policy target is emitted.

## Metadata and runtime controls

`metadata_factory.py` accepts a ROOT-registered control file with an observed phase first/deadline no longer than 7200 seconds and an explicit operator end. It checks the metadata-only teacher profile admission, selection lineage, sorted unique signed-int64 protected aliases and source manifest. It emits a canonical root-pool and immutable v2 registration; existing pool bytes may only be reused if identical. It never loads parent weights.

`run_collection.py` requires that registration, validates the exact frozen helper/candidate/profile/selection/protected hashes, clean core commit, output location, one CPU core, 15 GiB cgroup budget, 256 MiB workspace free floor, 128 MiB RAM-output cap, and original phase/operator clocks. It writes fsynced `events.jsonl` and sorted unique per-search signed-int64 alias sidecars (each at most 8 MiB) before recording their references. The final receipt binds source, profile, pool, protected inventory, events, and sidecar hashes. Output is publish-once under `/dev/shm/harbichess-ownq-v2/{seed}`.

## Verification status

Synthetic collector tests: `/workspace/HarbiChess/.venv/bin/pytest -q test_collector.py` (5 passed). `py_compile` passed. The 4096-row pool projector was run metadata-only for both seeds; each passed legal-history/FEN/digest validation. No actual parent inference/search/self-play/training or strength games were run by this agent. ROOT must freeze current source hashes, provide the protected alias manifest and fresh clock control, independently validate the registration/converter, then decide whether to launch.
