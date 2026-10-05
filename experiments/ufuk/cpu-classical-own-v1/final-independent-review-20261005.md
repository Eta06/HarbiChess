# Independent review: classical own-experience proposal

Review scope: read-only inspection of the prospective scratch pipeline, including its actor proof, offline trainer, transfer guard, resource guard, search profile, and arena audit. No production actor, fit, search profile, or match was run.

## Findings

- The same-seed proof-to-collection transfer is now fail-closed. The only permitted config differences are the four literal fields `max_actions`, `epoch_id`, `original_first_epoch`, and `original_deadline_epoch`; extra/missing keys and all other changed values fail. The proof config file SHA is tied to the qualification receipt. All three proof journals must embed that exact config and digest and independently replay before the 16,384-action collection can start.
- The actor qualification and offline whole/pause/resume qualification now check their original deadlines after the independent audit. The collection controller replays every segment and checks the same original deadline after each replay and at final audit. Its helper closure includes the segment runner, journal, learner, runtime, and transfer code.
- The repaired runtime imports the exact pinned clean-source cgroup budget helper. It preserves inactive-file eligibility, physical cgroup charge, and OOM-event checks, as well as disk floor, recursive artifact ceiling, and hard deadline.
- The learner uses only completed own-game WDL and own Q512 root values. Active/capped tails stay UNKNOWN; no SF labels enter training. Full-history protected-position exclusion applies to whole games, and the train/validation split is trajectory-based. The README correctly describes this as internal diagnosis, not position/source-independent validation.
- I ran the focused local suite from the proposal directory: `/workspace/HarbiChess/.venv/bin/pytest -q test_runtime_v2.py test_transfer.py test_pipeline.py test_native.py test_value.py` — **23 passed in 0.81s**. These are pure/synthetic tests; they do not qualify real actor speed, fit quality, or strength.

## Remaining gates and limitations

- The 18-board static-evaluation microbenchmark does not establish search throughput. At 16,384 actor actions and 512 nodes/action, the upper bound is 8,388,608 Qsearch nodes. Full-history repetition checks can make long trajectories slower. The registered 24-move profile must pass before collection is considered feasible.
- `arena/profile.py` lacks a final `time.time() < end` check after the last search and before it writes a PASS receipt. A search could pass its last inner guard before the deadline and finish after it. It also uses overwrite-capable `write_text` for its result. Add an outer deadline check and publish-once output before relying on this qualification.
- The search/evaluator is a handcrafted prior plus 18 learned scalars and fixed Q512 search. No evidence yet shows it improves over the prior or reaches the E8/SF gates. The proposal makes no novelty claim and appropriately distinguishes this candidate from NNUE.
- Search-root Q is a state-level training target even when epsilon exploration played a different move; the pipeline makes no trajectory importance-correction claim. Completion-conditioned examples and same-source trajectory validation remain biased/limited as documented.
- Arena max-ply caps score as protocol draws and are separately bounded by the cap gate. Stockfish receives a `nodes=512` request, but actual nodes may exceed the nominal cap; this is not equal-compute evidence.
- No actual actor proof, 16,384-action dataset, offline proof/fit, 24-move profile, 160-game audit, or strength result has run. All remain required before any strength claim.
