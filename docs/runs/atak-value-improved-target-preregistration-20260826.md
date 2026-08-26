# ATAK value-improved target pre-registration — 2026-08-26

## Frozen objective and baseline

ATAK starts a new iteration after closing V7 and its 600-game evidence set. No V7
checkpoint, replay, arena game, or threshold is reused. Both ATAK arms start from the
unchanged HAZIRAN champion model (`5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`).
The primary objective is a materially higher expected score, while retaining the
repetition and decisive-game behavior established by V7 where possible.

## Frozen target treatment

The control target is normalized MCTS visit count. The treatment starts from that
same distribution after the existing repetition-safe transformation and applies

`logit(a) = log(N(a)) + clip((Q_shrunk(a) - V(root)) / 0.10, -1.25, 1.25)`

where

`Q_shrunk(a) = (N(a) * Q(a) + 8 * V(root)) / (N(a) + 8)`.

This is target-only: it does not alter legal moves, MCTS traversal, root noise,
temperature, selected moves, game outcomes, or the established repetition redirect.
The eight-visit shrinkage and 1.25-logit cap keep sparse/noisy action values from
overriding search evidence. Target schema 8 identifies newly generated shards.

## Frozen replay and training budget

Two independent learners use matched, entirely fresh self-play generated after this
pre-registration. The matched seed is intentional: because the treatment is applied
after move selection, it isolates target quality on equivalent positions.

- Run seed: `2026082613`.
- 96 games per arm, 96 workers, 64 MCTS simulations, maximum 256 plies.
- 30 exploration plies and 0.25 ms inference batching wait.
- No continuation replay and no prior-generation replay.
- Same HAZIRAN champion initialization for both arms.
- 500 configured training steps, batch 64, validation every 10 steps.
- Early stopping patience 12 evaluations and minimum validation delta 0.001.
- Checkpoint selection will screen all meaningful validation checkpoints in a small
  same-generation arena; minimum validation loss alone will not choose the model.
- Treatment parameters, exposure, replay size, and training duration will not be
  changed after observing results.

## Frozen fresh arena

The selected control and treatment checkpoints each play the unchanged champion on
the same wholly new set of 200 color-balanced games (100 opening pairs):

- Arena seed: `2026082614`.
- Opening plies: 12.
- 32 MCTS simulations, maximum 256 plies, 96 workers.
- 0.25 ms inference batching wait.
- Paired bootstrap: 50,000 samples, seed `2026082615`.

The control/treatment checkpoint screens use a separate seed and are not included in
the final gate. The 200-game sample will not be extended after results are observed.

## Frozen promotion evidence

Advancing to promotion or a new generation requires all of the following:

1. Treatment minus matched control expected-score difference is at least +4.0
   percentage points and its paired two-sided 95% lower bound is above zero.
2. Treatment expected score against the champion exceeds 50%, with its ordinary 95%
   confidence lower bound above 50%.
3. Avoidable-threefold point estimate is no worse than control and its paired
   one-sided 95% upper bound is at most +5 percentage points.
4. Win-rate point estimate is above control and its one-sided 95% lower bound is no
   worse than -2 percentage points.
5. Conditional decisive score does not regress relative to control.

The existing paired gate computes conditions 1 (interval only) and 3–5. The +4-point
minimum effect and absolute champion condition are checked explicitly in the final
report. Failure leaves the champion and generation unchanged; thresholds, sample
size, and treatment will not be tuned on this arena.
