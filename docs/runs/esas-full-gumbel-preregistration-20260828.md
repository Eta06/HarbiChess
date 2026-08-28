# ESAS Full Gumbel Search Preregistration

Date: 2026-08-28  
Status: frozen before implementation and results

## Hypothesis

The existing root-only Gumbel helper is not Full Gumbel MuZero. An exact
shared-tree implementation of root Sequential Halving, interior deterministic
improved-policy visitation, and mixed-value completed-Q may allocate the same
clean search budget more reliably than legacy PUCT despite the weak value head.

This is an allocation test, not an assumption that Gumbel's policy-improvement
guarantee applies: that guarantee requires correctly evaluated action values,
which HarbiChess has not established.

## Frozen mechanism

- Shared search tree and exactly one leaf expansion per simulation.
- Root selection follows Mctx's considered-visit Sequential Halving schedule.
- Interior selection approximates `softmax(prior_logit + completed_q)` through
  deterministic visit-frequency matching.
- Unvisited Q values use Mctx's mixed value completion.
- Completed Q is min-max rescaled, then multiplied by
  `(50 + max_child_visits) * 0.1`.
- Maximum considered root actions: `16`.
- Gumbel scale: `0.0`, because this is deterministic clean evaluation in a
  perfect-information game.
- No Dirichlet noise, FPU treatment, continuation/repetition target, learner
  update, or replay generation.

## Frozen qualification workload

- Model and checksum: KOPRU baseline,
  `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`.
- Budgets: `64`, `128`, `256` simulations.
- Arena: 32 fresh deterministic opening pairs per budget, color-swapped for 64
  games per arm, against the raw network.
- Max plies: 256; max-ply remains behavioral telemetry and is not a value label.
- Tactical suite: the existing eight rule-verified cases.
- Worker/inference settings: 24 workers, 250 microsecond batch wait.

## Gates

Use the already frozen ESAS system-teacher gate unchanged:

1. Both 128 and 256 score strictly above 55%.
2. The paired 256 lower confidence bound strictly above 50%.
3. The 256 score not more than two percentage points below 128.
4. At least two additional tactical solves over raw at 256.
5. No tactical solve-count regression or lost solved case from 128 to 256.
6. Decisive score at least 50%.
7. Max-ply and threefold rates no more than ten percentage points worse than
   raw control.

All gates must pass. Constants, sample size, and thresholds will not be changed
after observing this run. Failure keeps learner/latest, generation, and
promotion blocked and ends this allocation branch.
