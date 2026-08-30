# DEVRIYE continuous learner pilot preregistration

## Purpose

Test continuous policy iteration without asking each small update to defeat the release champion.
The MIHVER nonlinear invariant checkpoint is the immutable starting point. The learner carries both
latest weights and optimizer state across three updates; only a catastrophic gate failure rolls the
state back to the preceding accepted update.

This is a small fixed-replay pilot, not a promotion run. No large self-play generation is authorized
before the complete chain passes.

## Frozen data and teacher

- Corrected, trajectory-disjoint KOPRU train/validation replay.
- Three updates, each using fresh non-overlapping stratified replay positions.
- Per update: 96 train and 48 validation Full Gumbel targets.
- Clean Full Gumbel teacher, 256 simulations, top 16 actions, zero Gumbel noise.
- Teacher is always the latest accepted network, never the release/random baseline.
- Rolling policy buffer retains the latest two 96-row train target generations.
- WDL batches retain the corrected replay outcomes and mixed sampling: half outcome-balanced, half
  natural game-balanced.

## Frozen learner

- 40 joint head steps per update, batch 64, Adam `1e-4`, gradient clip 5.
- Trainable: release policy head plus MIHVER invariant/global WDL head.
- Frozen: release trunk, legacy release WDL head, spatial value tower, and auxiliary material head.
- Optimizer state continues across accepted updates.

## Per-update gates

- Fresh teacher validation policy CE improves by at least `0.01`; top-action agreement does not
  regress.
- WDL micro and macro CE may not regress by more than `0.01` from the previous accepted network;
  expected-score Pearson may not regress by more than `0.02`; ECE-10 stays at most `0.12`.
- Absolute MIHVER WDL floors remain: micro CE <= `0.99968`, macro CE <= `0.99890`, Pearson >= `0.20`,
  both outcome margins >= `0.03`.
- Deterministic material MAE <= `0.05`, Pearson >= `0.80`, and predictions remain exactly unchanged.
- Fixed continuation set: mean Spearman >= `0.05`, verified-top agreement >= `0.34375`.
- Full Gumbel 256 tactical solves at least `5/8` and loses none of MIHVER's solved cases.
- Four fresh color-balanced opening pairs at 32 simulations against the previous latest network;
  score below `0.375` is catastrophic and triggers rollback.

## Chain gate

All three updates must be accepted. A final fresh eight-pair, 64-simulation arena compares update 3
against the immutable MIHVER start; score must be at least `0.50`. Final policy imitation, WDL
Pearson, continuation ranking, and tactical strength must be no worse than their accepted chain
floors. These small arenas are safety/direction evidence only and cannot authorize promotion.

Thresholds, sample sizes, seeds, replay exposure, learning rate, and run length may not change after
results. Passing authorizes production continuous-generation/promotion integration; failing keeps
generation closed and starts a causal audit without relabeling the run.
