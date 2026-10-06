# Own-search collection prototype (unlaunched)

`collector.py` has a pure collection API. `run_collection.py` is a registered, CPU-only production entry point; it is not run by this prototype. No parent inference, search, selfplay, or training was performed here.

Expected input registration pins the clean core commit, producer source hashes, Python/native/evaluator/prior/extension/search helpers, same-seed teacher-bootstrap candidate and canonical contract SHA, 4096-row TRAIN-only root-pool SHA, protected-alias binary SHA, output path under `/dev/shm/harbichess-ownq-v1`, one CPU core, and immutable start/deadline. The candidate must be the per-seed `whole/candidate.pt`, 256-update teacher-bootstrap phase. Fixed search is 8192 nodes, qdepth 2, max depth 8; fixed roots are 128 selected outcome-blindly from the TRAIN pool, with 16 plies per start and a target of exactly 1024 eligible search rows. Insufficient rows fail without VAL autofill.

Each `search_row` stores the full root FEN/prefix and current full UCI history before the selected move, mover, raw mover-perspective search Q and clipped target, selected move, node/evaluation/depth counts, and sorted unique current-piece-placement aliases for every actual static-model input. `mate_range_score_returned` is only an alpha-beta score-range flag, not a proof certificate. Completed own games add mover-perspective WDL; caps and row-budget tails remain WDL UNKNOWN. Any trajectory that reaches a protected state or whose search evaluates a protected alias is excluded in full from learning. A protected-search row records `selected_action_played=false`; its proposed move is not pushed. No policy target or Stockfish label is generated.

Events are fsynced as JSONL. Alias sidecars are sorted unique signed int64 values per search row, written before the event reference, in chunks no larger than 8 MiB; the final receipt binds each chunk SHA. Root/converter review and a new immutable registration are still required before any collection run.

Synthetic checks use `/workspace/HarbiChess/.venv/bin/pytest -q test_collector.py`. They do not qualify the production runner or establish strength.
