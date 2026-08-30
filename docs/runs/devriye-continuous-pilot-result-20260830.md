# DEVRIYE continuous policy-iteration pilot result

## Decision

The rolling latest-network mechanism now works and completed three accepted
updates, but the immutable MIHVER final chain gate failed by small cumulative
WDL/continuation drift. Continuous production generation, release promotion,
and champion replacement remain blocked. The three checkpoints are retained as
diagnostic artifacts, not qualified latest/release models.

Primary artifact:
`artifacts/runs/devriye-continuous-pilot-20260830-13/result.json`

## What now works

- Versioned Full Gumbel latest-network replay shards with checkpoint and commit
  provenance, checksums, phase-balanced continuation starts, and correct unknown
  max-ply value masking.
- Correct Full Gumbel soft replay targets. Behavior selection can differ from
  sparse root visits without violating replay schema semantics.
- Rolling two-generation policy and value replay, latest checkpoint teaching,
  non-overlapping searched targets, and dashboard self-play/training status.
- 768 train plus 192 validation Full Gumbel-256 targets per update. This fixed
  the measured 96-row policy generalization failure without increasing the
  40-step learner duration.
- 96 continuation games per update with 24 workers and a 24-terminal-game floor.
- Independent earliest policy/value checkpoint selection, composed only across
  disjoint heads; frozen parameters remain bitwise stable. Adam moments reset
  after composition because their global bias-correction step cannot be spliced.
- Value validation at single-step cadence until its first healthy checkpoint.
- Honest distinction between update rollback and a final-chain rejection in the
  dashboard.

## Fresh three-update result

All three updates passed their policy imitation, relative WDL, material,
continuation floor, Full Gumbel tactical, and catastrophic mini-arena gates.

| Update | Policy step | Value step | Policy CE before -> after | Top agreement | Terminal games | Tactical | Mini arena |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 1 | 3.07253 -> 3.05345 | 4.17% -> 18.75% | 27 | 5/8 | 1-6-1, 0.500 |
| 2 | 10 | 1 | 2.96488 -> 2.94912 | 16.67% -> 18.23% | 36 | 5/8 | 1-7-0, 0.5625 |
| 3 | 10 | 1 | 2.83933 -> 2.81741 | 15.63% -> 19.79% | 25 | 5/8 | 2-5-1, 0.5625 |

The final 16-game Full Gumbel-64 arena against immutable MIHVER was `2-13-1`,
expected score `0.53125`, decisive score `0.6667`, and threefold rate `0`. Search
strength therefore did not trigger the rejection.

## Final cumulative rejection

| Metric | MIHVER | Update 3 | Delta |
|---|---:|---:|---:|
| WDL micro CE | 0.912848 | 0.913124 | +0.000275 |
| WDL macro CE | 0.937718 | 0.937870 | +0.000152 |
| WDL Pearson | 0.452612 | 0.451688 | -0.000924 |
| Continuation Spearman | 0.073161 | 0.071382 | -0.001778 |
| Continuation top agreement | 37.50% | 37.50% | 0 |

Per-update tolerances allowed tiny changes that accumulated past the exact final
no-regression guardrails. The thresholds were not relaxed and the positive arena
did not override them.

## Root-cause sequence

1. Standard-start 12-game replay yielded only 2 known outcomes. Phase-balanced
   continuation starts corrected terminal efficiency without inventing draws.
2. Full Gumbel behavior exposed two implementation faults: replay used sparse
   visits instead of the improved soft target, and nearly equal distributions
   could produce a tiny negative floating KL. Both were corrected and tested.
3. Natural short-window WDL sampling became outcome-imbalanced; uniform W/D/L
   then distorted class priors. Fixed 25/50/25 fresh sampling removed that
   variance but did not solve low independent-game count.
4. Scaling to 96 continuation games supplied 25-36 independent terminal games
   per update. Scaling teacher labels from 96 to 768 fixed policy transfer.
5. A single 10-step checkpoint cadence coupled independent policy/value heads.
   Head-wise selection plus value step-1 validation fixed local transfer and
   passed a cached full tactical/continuation/arena qualification.
6. Three fresh updates revealed the remaining blocker: cumulative
   stability-plasticity drift in the existing MIHVER value head/objective.

## Rejected follow-up mechanisms

- MIHVER-output distillation weights `0, 0.25, 1, 4, 16`: no joint old/fresh
  pass; the anchor gradient is zero at the initial function.
- Single-generation value batches `64, 256, 1024, 2048`: larger batches improved
  fresh CE but not the complete old/fresh CE+Pearson gate.
- Pooled 88-game batches `1024, 2048, 4096`: fresh CE improved, but at least one
  exact old metric and fresh Pearson still regressed.
- Explicit micro+macro CE plus Pearson-loss weights `0, 0.25, 1, 4`: no arm
  generalized to all fixed gates.

More sampling-weight, duration, batch-size, or same-head loss tuning is therefore
not justified.

## Next technical decision

The highest-probability next experiment is a separately preregistered stable-base
plus plastic-residual value representation: freeze qualified MIHVER value as the
non-forgetting path, train a zero-initialized state-dependent residual on pooled
fresh outcomes with historical rehearsal, and require simultaneous old/fresh
WDL, continuation, Full Gumbel tactical, and search-strength improvement. Policy
and search allocator changes are not indicated by this run.

## Runtime and storage

The fresh `-13` run took 1,889.24 seconds (31.49 minutes). Its artifact directory
is 52 MiB. All DEVRIYE pilot directories `-05` through `-13` total approximately
141 MiB and remain reproducible diagnostic evidence.
