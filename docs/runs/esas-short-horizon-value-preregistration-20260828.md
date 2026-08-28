# ESAS short-horizon value preregistration (2026-08-28)

## Hypothesis

The frozen release network cannot provide useful non-terminal leaf ranking
because its WDL expected value is nearly constant. Final outcomes alone are too
sparse and high variance in the current replay. A separately headed short-horizon
value target can shape the shared trunk without changing the semantic meaning of
the main WDL head.

This follows KataGo's exponentially averaged future-MCTS-value mechanism. It is
not a material heuristic, a replacement WDL label, or a policy-target transform.

## Frozen data and targets

- Replay: all train and validation shards from
  `kopru-qualified-replay-20260828-01`
- Baseline SHA-256:
  `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`
- Main policy target: stored clean visit distribution, unchanged
- Main WDL target: final outcome only; 9,216 max-ply rows remain masked
- Auxiliary target: one scalar head predicting the signed exponential average
  of future stored root search values
- Recurrence for a row at ply `t`:
  `a_t = (1-lambda) q_t - lambda a_(t+1)`
- `lambda = 0.8`; at a known terminal tail use the row's side-to-move outcome;
  at a max-ply tail use the final stored root value
- Every perspective flip is explicit and unit-tested

Stored root value has only 0.294 correlation with known outcomes. It is therefore
never mixed into the WDL label and the auxiliary head is not used directly by
search in this experiment.

## Frozen arms

Both arms start from the exact baseline and use the same shuffled batches,
AdamW `2e-4`, batch 64, two replay-equivalent epochs (474 steps), seed
`2026082869`, and validation every 79 steps.

1. Control: existing policy + masked terminal-WDL loss.
2. Auxiliary: the identical loss plus scalar Huber short-horizon loss with
   weight 0.25.

The auxiliary head is discarded at export. Only the standard policy/WDL network
is evaluated. No arm may receive additional steps.

## Frozen gate

The auxiliary arm is selected only if, relative to both baseline and matched
control:

1. known-outcome validation WDL cross-entropy does not regress by more than 1%;
2. validation expected-value/outcome Pearson correlation improves by at least
   0.05 over baseline and is no worse than control;
3. expected-value standard deviation is at least 0.02, excluding another
   constant-value solution;
4. validation policy cross-entropy is no worse than control by more than 1%;
5. raw tactical solve count is no lower than baseline;
6. 128- and 256-simulation tactical solve counts are no lower than the baseline
   counts 7/8 and 7/8, and no case solved by baseline 128 is lost at auxiliary
   256;
7. losses and gradients are finite and maximum gradient norm is at most 5.0.

Passing authorizes a fresh rerun of the unchanged ESAS system-teacher
qualification with the auxiliary-trained standard network. It does not authorize
replay generation, continuous learner, arena promotion, or release promotion.

Failure rejects this one-head/lambda/weight contract. Its parameters will not be
tuned on the same replay.
