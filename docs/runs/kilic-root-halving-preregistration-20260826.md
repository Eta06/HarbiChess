# KILIC root sequential-halving pre-registration — 2026-08-26

## Frozen objective and baseline

KILIC is a new iteration after ATAK failed its strength gate. It does not reuse V7 or
ATAK replay, checkpoints, arena games, or thresholds. Both arms start from the
unchanged HAZIRAN champion model with SHA-256
`5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`.
The objective is a larger expected-score improvement from fewer, higher-confidence
policy/search changes while preserving repetition and decisive-game behavior.

## Frozen fixed-compute treatment

Both arms spend 65 neural evaluations per ordinary non-terminal root, counting the
initial expansion. The control runs the existing 64-simulation MCTS. KILIC allocates
the same budget as follows:

1. 32-simulation broad noisy-root MCTS (33 evaluations including root expansion).
2. Top four visited actions each receive a forced deterministic child search with
   three simulations (4 × 4 = 16 evaluations including child roots).
3. The best two first-round actions each receive a seven-simulation child search
   (2 × 8 = 16 evaluations including child roots).

Total: `33 + 16 + 16 = 65`, equal to the control's `1 + 64 = 65`.

The target changes only when the same winner ranks first in both forced rounds and
beats its runner-up by at least 0.05 root value in each round. When qualified, 35% of
the other top-four policy mass moves to the winner. When uncertain, the initial visit
policy is unchanged. Target schema 9 records whether a root was adjusted and both
confidence margins. Existing legal-move, terminal, perspective, repetition-defense,
and avoidable-threefold transformations remain active after the refined root search.

No ATAK value-temperature target is combined with KILIC. The 0.05 confidence margin,
35% transfer, top-four/top-two allocation, and search budgets will not change after
fresh generation begins.

## Frozen replay and learner budget

- Fresh run seed: `2026082617`.
- Shared deterministic split namespace: `kilic-fresh-split-20260826-01`.
- 96 games per arm, 96 workers, total root budget 64, maximum 256 plies.
- 30 exploration plies and 0.25 ms batching wait.
- No continuation replay and no earlier-generation replay.
- Same champion initialization for both arms.
- 500 configured training steps, batch 64, validation every 10 steps.
- Early-stopping patience: 12 evaluations; minimum delta: 0.001.
- All meaningful validation checkpoints receive a 32-game same-generation screen on
  seed `2026082620`; minimum validation loss alone does not select the checkpoint.
- The screen is excluded from final evidence.

The split namespace makes the same game indices train or validation in both arms,
without conflating their artifact/run identities. Treatment trajectories may differ
because a qualified refined root can alter move selection; this is part of the search
intervention rather than replay leakage.

## Frozen fresh arena and gate

Selected control and KILIC checkpoints each play the champion on exactly 200 wholly
new color-balanced games (100 opening pairs):

- Arena seed: `2026082618`.
- Opening plies: 12.
- 32 arena MCTS simulations, maximum 256 plies, 96 workers.
- Inference batching wait: 0.25 ms.
- Paired bootstrap: 50,000 samples with seed `2026082619`.
- The 200-game sample will not be extended after observing results.

Promotion/new-generation consideration requires every condition:

1. KILIC minus control expected-score estimate is at least +4.0 percentage points
   and its paired two-sided 95% lower bound is above zero.
2. KILIC expected score against the champion exceeds 50%, with its ordinary 95%
   confidence lower bound above 50%.
3. Avoidable-threefold point estimate is no worse than control and its paired
   one-sided 95% upper bound is at most +5 percentage points.
4. Win-rate point estimate exceeds control and its one-sided 95% lower bound is no
   worse than -2 percentage points.
5. Conditional decisive score does not regress relative to control.

The final report also records root evaluation/adjustment coverage, adjustment margin,
terminal distribution, throughput, early-stopping reason, checkpoint screen, and
model hashes. Failure leaves the champion and generation unchanged; no threshold,
sample-size, exposure, or temperature tuning is permitted on this evidence set.
