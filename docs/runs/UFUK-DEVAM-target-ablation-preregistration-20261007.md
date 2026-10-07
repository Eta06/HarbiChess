# UFUK-DEVAM: own target ablations, 7 October 2026

This prospective registration follows the completed own-Q v2 development screen. That method failed in both seeds: Stockfish nominal512 scores .03125/.0625, own exact teacher-parent scores .40625/.5625. Independent fullhistory audit and six actual packet replays passed; these are integrity checks, not learning strength. Both teacher parents themselves scored 0/.03125 against the same Stockfish control. No model is promoted and the approved ONE blind confirmation has not been drawn.

## A: same-data TD(lambda) controlled target ablation

Hypothesis: moving the frozen own-search values along an alternating-mover legal trajectory toward actual terminal results can produce better learning targets than isolated pre-action Q. Fixed lambda=.5 uses actual own terminal WDL where available; UNKNOWN episodes use their last logged pre-action Q bootstrap. No cap is labelled a draw and no unlogged post-action value is invented. This remains standard TD learning, not a novelty claim.

Reuse exactly the same 1,024 eligible rows per seed (20262905/20262906), exact 256-step teacher-trained initializer weights, architecture, human prior, 64 updates, batch256, Adam(.001,.9,.999,eps1e-8), clip5, MSE-tanh value, deterministic CPU float64. No new teacher calls. Change ONLY target method/native phase. Teacher-bootstrap ancestry remains disclosed. It is a new weights-only initialization with fresh optimizer/global/private RNG, never full resume of the old Q or teacher training.

Frozen runtime: `experiments/ufuk/continuation-20261007/tdlambda-training-v1/source`; ROOT executor `execute_tdlambda_chain.py`. Distinct typed dataset/provenance/contract/native schemas reject old-phase resume. Before production64: actual whole8/pause4/fresh-process-resume8 all-payload bit equality and six actual fresh strict-native loads. Then actual fresh64 plus strict loads at0/64; do not initialize64 from proof-final8.

Each seed has CPU1/3, an original600-second conversion, original600-second proof and original1800-second fit, bounded within a3000-second ROOT-owned stage and operator ceiling1791448916.685839. Sources/inputs/phase clocks are SHA-bound before their execution. RAM outputs only; shared memory guard15GiB, disk floor256MiB. Exceptions preserve the failed attempt; no budget extension or same-phase retry.

Use newly typed variant admission and unchanged original `de53c147...` search (512 nodes/q2/max8) for the known8, color-paired160-game screen, including child-E8, child-exact-parent, child-SF, parent-SF and E8-SF for both seeds. Numeric gates are unchanged: direct E8>.60, direct parent>.60, SF>=.25, paired SF gain over E8>.10, gain over parent>0, caps<=.05; median wall ratio<=1.10. Both seeds must pass. Known8 is development selection, not independent proof. Only an eligible frozen candidate can reach the previously approved ONE new48-root-per-seed confirmation, with all original bootstrap gates AND the preregistered additive exact betting bounds. Do not pool seeds to rescue a failure.

## B: completed self-play terminal targets

Hypothesis: 85–90% UNKNOWN rows in the short16-ply collection weaken reward propagation. A separate prospective collection plays trajectories to actual terminal states (up to400 total plies) with the same teacher parents, deterministic TRAIN start order and 8192/q2/max8 search. Choose the first1,024 chronological eligible rows from CLOSED terminal games only, without outcome/rating filtering; finish the current episode. UNKNOWN caps and any protected-input episode are retained as raw failures and completely excluded from training, never converted to draws. Fixed128 starts, actor-row ceiling4096, collection7200 seconds per seed, event128MiB/alias128MiB, same operator/memory/disk guards. Failure to reach1,024 rows is a collection failure; do not grow ceilings after observing it.

Distinct MC dataset/native phase uses true own-game mover WDL (lambda1/Monte Carlo), fixed64 updates and the same actual proof/fresh-fit and strength requirements as A. No new Stockfish labels. Standard terminal-value self-play is prior art. Synthetic fixture qualification is not real collection or strength evidence. ROOT will pin the complete frozen runtime and actual original collection clocks before execution.

## Separate search diagnosis

PVS/history ordering/full-history TT and checked-quiescence handling can improve the baseline independently of learning. Compare the same human prior under new search and original de53 on paired known positions/games. Label any improvement SEARCH-ONLY. A later own-learning test must compare child and exact frozen parent under the SAME search, preserve all strength gates and log source/position exposure. Search improvement alone cannot complete the user goal. No rules/world-model or novelty success is claimed.

CPU only; no GPU/SSH/paid resource use. Preserve all prior failures/checkpoints. Important results go to Git/Space; models/native/replay bytes go to versioned public-readback-verified Release assets.
