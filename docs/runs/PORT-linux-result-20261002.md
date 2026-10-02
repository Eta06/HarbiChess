# PORT Linux result — 2026-10-02

## Decision

Linux inference, real optimization, fresh rolling self-play and generation-boundary full training
resume passed their execution preflight. **Strength/generalization improvement is not established**:
latest versus initial scores 50%; frozen held-out WDL CE worsened 1.145656 → 1.528078. Stockfish
19 at only 32 nodes/move still scores 13 wins/3 draws against latest. No champion promotion or
long training was started after this evidence. DENGE remains historically failed.

This is a bounded technical handover round, not completion of the strong-chess objective.
Next work is profiling and fixed-wall-budget search/data-ratio ablation with a larger game-family
held-out set, not more heuristics or unmeasured long training. User authorization remains the
source of authority; no paid resource purchase or schedule was created.

## Access and source

Repo Eta06/HarbiChess: authenticated connector and CLI user `Eta06`, admin/push permissions.
Initial work HEAD/main `f4bdc85638b18cb313278567ceabf99fb21798a0`;
push dry-run was up-to-date; no access-test commit. Git author/committer independently verified
as Emir Tunahan Alim <emrtnhalim@gmail.com>; no co-author trailers. Push identity and commit
identity are distinct even though this session uses the user's GitHub account.
Space page_bfc28b10daa081919e1650eca18c61c7 read/write permissions checked without content changes.

Initial training and benchmark implementation snapshot: `6b1ad06b202095e42332b46d9d628d4dc1638e6d`.
Resumed training snapshot is recorded in each checkpoint (documentation-only changes at restart).
Later relocation, parity-extra and KL diagnostic commits are separate; no old checkpoint provenance
was rewritten. Arena implementation was a9463f6 throughout these games; early arena reports
captured HEAD at completion while documentation/resume changes proceeded. The later source-capture
fix records HEAD before an arena starts. This provenance limitation is preserved in the raw records.

## Hardware, compute and dependencies

Linux x86_64, AMD EPYC 9V74; 5 visible CPUs but cgroup quota 4 CPUs; memory.max=16 GiB,
no swap, about 30 GiB free initially. NVIDIA devices absent; `torch.cuda.is_available()=False`.
Torch 2.14.1+cpu; MLX/MLX CPU 0.32.1; Python 3.12.14. Existing .venv retained, only missing
optional packages installed and locked. Apple MLX marker retained; CUDA path has no device test.

Training sessions: wall 118.548 s, process CPU
117.861 s, max process RSS 1203380 KiB (~1.15 GiB).
Self-play 93.097 s, optimizer steps 2.697 s.
Preparation/checksum/checkpoint overhead is included in session wall time. CPU/wall is about
one core; actors are Python threads and GIL/rule/encoding work remains a speed bottleneck.
This is below the four-core quota; utilization is not treated as a success metric.
No new paid resources; the allocation's billing tariff was unavailable, so no currency cost is invented.

## Parity, rules and training semantics

Shared schemas unchanged: `{'replay': 2, 'encoder': 1, 'action': 1, 'target': 12}`. Encoder NHWC 8×8×104, policy 4672 actions,
STM WDL order win/draw/loss, search value P(win)-P(loss); legal masking and alternating backup
preserved. Base/invariant/MIHVER/DENGE forward paths ported, including count-scaled and nonlinear
global value heads, spatial residuals, material head and plastic logit scale. Unsupported experiment
heads fail strict conversion rather than disappear. Both PUCT and Full Gumbel real backend mate/sign
cases pass for white/black; existing castling, en-passant, promotion, history/repetition tests pass.

Archived MIHVER SHA-256 `6c535585e952b3d8ba5ff2331fe4dc992d94c586fdfa6a7c3c0cbc9e2d229dbb`;
archived failed DENGE update-003 `c9985d38a125f9505ef30d38a8acdd1a776c3f3c1d93fe620fef233f8e0bc1a2`.
Both independently matched NumPy NHWC and real MLX CPU to <1.2e-7 maximum absolute forward error.
Real framework forward/masked-policy tests cover all four paths; losses and gradients also match.
Apple Metal/BF16 behavior/performance was not tested. Linux MLX CPU tests do not close that gap.

Learner loss matches MLX legal-masked policy CE and weighted terminal WDL CE. Unknown capped
outcomes use zero value weights. **Optimizer behavior differs deliberately**: PyTorch AdamW has
bias correction; existing MLX AdamW defaults false. No equal-step or cross-framework optimizer
resume equivalence is claimed. Version 1 conversion is FP32 weights-only warm start; old files
remain unchanged. Full training resume is a different PyTorch checkpoint schema, containing
optimizer, sampler/torch/CUDA RNG, config/runtime, relative replay checksums and generation/game
cursor. Torch runtime/device/threads/determinism must match. Full run relocation and early-v1
run_id migration are tested; single-worker CLI resume is bitwise identical to uninterrupted training.
Multiple-worker batching has no unconditional bitwise replay claim; conflicting partial replay fails.

Old replay actually restored and checksum-verified: KILIC validation shard
`905c9296dd2d7add08b5525055f80b34c3656c23531c1784190c4c6129cca10e`;
target schema 9, 3 records/3 games, rules validated and training batch constructed without conversion.

## Fresh loop

Frozen qualified MIHVER warm start, all-parameter AdamW lr=2e-4/decay=1e-4; seed 20261002;
3 generations × 8 games, 16 Full Gumbel simulations (Gumbel=1), cap 192 plies, 4 actors,
1 inference/compute thread, 32 updates/generation, batch 32, game-balanced sampling,
three-generation rolling replay. Last complete game per generation is validation and never trained.
No oracle labels, second clean search, repetition transform, retrospective gates or promotion.

Actual: 24 games, 3089 unique generated records, 20
terminal-observed games, 2321 known value rows, 96 updates,
3072 sampled update rows. Sampling counts are not unique generated positions.
The process exited after generation 1 and a new process restored generation 1 to finish 2/3 with
optimizer continuation and game cursor 24. `session-generation-1.json` preserves the first session.
Generation self-play throughput 588.09/560.50/545.40 inference positions/s;
aggregate 562.04/s including Python search/encoding/queue overhead.
Loss on the fixed first training sample decreased 4.033412 → 3.265749. This is execution/fit evidence.

Frozen same held-out comparison: 3 whole games, 311 rows, only 119 known terminal targets.
Initial total/policy/WDL CE [3.8745765686035156, 2.7289206981658936, 1.1456559896469116]; latest [4.220728874206543, 2.6926512718200684, 1.5280778408050537].
Policy CE improved slightly; WDL CE deteriorated. The small terminal-selected holdout has
limited independence/coverage and cannot prove general strength. The final capped game remains
masked; it was not relabeled as a draw to improve metrics. The loop remains a research learner.

## Speed

Warmed masked backend: 5 warmups, 30 measured repeats, fixed initial position, encoding outside
timing; input creation, forward, legal gathers and host outputs included. FP32, same weights.
Single-thread setup selected and frozen before self-play. Benchmark was rerun in isolation after
implementation commits; concurrent development measurements were discarded as the final speed record.

| Threads | Batch | Positions/s | Mean batch ms | P95 ms |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1341.8 | 0.745 | 0.786 |
| 1 | 8 | 2535.2 | 3.156 | 3.313 |
| 1 | 32 | 2998.9 | 10.670 | 11.139 |
| 2 | 1 | 679.7 | 1.471 | 1.585 |
| 2 | 8 | 2203.5 | 3.631 | 3.878 |
| 2 | 32 | 2851.4 | 11.222 | 11.519 |
| 4 | 1 | 1029.6 | 0.971 | 1.072 |
| 4 | 8 | 2120.3 | 3.773 | 4.304 |
| 4 | 32 | 2749.3 | 11.639 | 13.882 |

These rates are not games/s, full search rates or a speed comparison with Apple Metal.

## Paired strength diagnostics

Eight fixed opening families × both colors; 16 games/control, identical opening histories;
seed 20261002, deterministic Full Gumbel=0, 16 sims/move, cap 192 plies. Stockfish 19 official
Linux universal binary, one thread, Hash=16 MiB, nodes=32/move, no pondering; binary SHA and
compiler details in evidence. Random chooses uniformly from legal moves. No-op tests initial vs
itself; latest-vs-initial separates learning from merely adding search. Openings, moves, outcomes,
color, time, evaluated positions and pair scores saved in JSON.

| Control | Latest/control W/D/L | Score | Caps | Pair bootstrap 95% | Wall s |
|---|---:|---:|---:|---|---:|
| latest-vs-initial | 2/12/2 | 0.50000 | 5 | [0.375, 0.625] | 56.43 |
| latest-vs-random | 16/0/0 | 1.00000 | 0 | [1.0, 1.0] | 18.29 |
| latest-vs-stockfish | 0/3/13 | 0.09375 | 0 | [0.03125, 0.1875] | 10.74 |
| noop | 3/10/3 | 0.50000 | 2 | [0.5, 0.5] | 65.08 |

Caps count 0.5 only for diagnostic scoring and are listed separately. Bootstrap intervals are
conditional on this fixed suite and can degenerate; Hoeffding pair intervals are also recorded
(latest vs initial ~[0.020,0.980], vs Stockfish [0,0.574]). Eight pairs do not establish Elo or
non-inferiority. Arena wall times include some overlapping verification work in the same allocation;
strength budgets are fixed nodes/simulations, so throughput is descriptive, not an isolated speed claim.

## Validation and resolved failures

Final full suite: **461 passed in 27.31 s, zero skipped/failed**; Ruff and compileall pass.
Initial no-MLX run had 65 collection errors and 11 pre-existing skips, preserved in evidence.
Adding actual Linux MLX CPU removed these limitations instead of adding new skips.
The first real full run had 454 passed/1 failed: KL diagnostic FP32 cancellation -9.8e-8.
The production diagnostic now clamps tiny negative roundoff and rejects nonfinite/<-1e-5 values;
probability-reference/shift-invariance regression added. Historical gate thresholds/results unchanged.
A temporary arena metadata commit left an unused capture (lint failure); its follow-up consumes
that capture. No game/search behavior changed. Final Ruff/compileall pass; source capture is metadata.

Primary mechanisms/limits and later-only Laya/dataset/stronger-student/LLM questions:
[PORT decisions](../research/PORT-decisions-20261002.md). Laya URL returned 404; identity not guessed.
No new large side project. Do not treat policy imitation, random-opponent wins, or high utilization
as stronger-than-teacher proof.

## Artifacts and recovery

[PORT Release](https://github.com/Eta06/HarbiChess/releases/tag/port-linux-preflight-20261002):
50 files, four complete model/optimizer checkpoints, six fresh replay shards, raw game/measurement
JSON, initial/final test failures and successes, source snapshots, source/model/data hashes and logs.
Raw archive SHA-256 `77846c8a557b301cdc4687f77c4c22e366036c39993e21c04152780ebca71235`.
[Member manifest](../research/PORT-artifact-manifest-20261002.json). Archive source checkpoint
metadata is retained; publication is not promotion. Originals and downloaded historical archives
were not deleted. Verify Release SHA256SUMS, extract to an empty directory, verify member hashes,
then use [runtime/resume guide](../PORT-runtime.md). All generated artifacts stay outside Git history.

| Checkpoint | Learner step | Model SHA-256 | Training source commit |
|---|---:|---|---|
| generation-000000 | 0 | `a155be94c96227815a2cac6f235c06622a5fb1df93a215b4ba391be500aaaa61` | `6b1ad06b202095e42332b46d9d628d4dc1638e6d` |
| generation-000001 | 32 | `2ec31c35c34d0b92e8a5848660f687af30da70314c8bff4caab3194a4beba4d5` | `6b1ad06b202095e42332b46d9d628d4dc1638e6d` |
| generation-000002 | 64 | `f79bde46ac9d4c096aa9e73e25ddadb3d5d24c4ef52c82627a64b6682568fc8d` | `e0d0f550b23bba5cd0c020fc15c7f54b0b1caf94` |
| generation-000003 | 96 | `e6920ef586260a5cca279413d952d839ce9a9ef62d9865c6d4c5af0e4b1e7f83` | `e0d0f550b23bba5cd0c020fc15c7f54b0b1caf94` |

[Atomic commit/hash/file inventory](PORT-linux-commits-20261002.md) includes all PORT files.
Author/committer are the requested identity, no co-authors, authorized CLI push to main.
Space summary is updated only after GitHub evidence is available. Final sync receipts and clean
Git status are reported in the closing commit inventory/chat.
