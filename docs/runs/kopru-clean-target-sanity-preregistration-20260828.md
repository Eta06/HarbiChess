# KOPRU clean-target sanity preregistration

The 96-game replay is retained as evidence but is not eligible for further learner training:
stored noisy targets agreed with the qualified clean teacher on only 6.25% of a fresh stratified
audit, with mean TV 0.490. Conservative visit pruning did not recover alignment (14.58% top-action
agreement and TV 0.474 on a controlled noisy-search replay).

This sanity run separates exploration from supervision without doubling search compute. MCTS runs
without root noise and its clean visit policy is stored as the training target. During the first 30
plies only, move selection samples from a 75% clean visit-policy / 25% Dirichlet mixture. The mixed
distribution is never written as the policy target.

Frozen settings:

- Run ID: `kopru-clean-target-sanity-20260828-01`
- Source model and depth-1 teacher: unchanged
- 48 fresh games, seed `2026082805`, 24 actors, 8 oracle processes
- 64 simulations, maximum 256 plies, validation fraction 25%
- Search root noise: off; selection Dirichlet alpha 0.3, fraction 0.25
- Continuation/repetition transforms, value-policy adjustment, and root-halving: off
- Learner: off until replay coverage and target-alignment gates pass

The existing replay coverage thresholds remain unchanged. A second 48-position alignment audit,
seed `2026082806`, must additionally satisfy all of these target-integrity gates:

- stored target vs clean teacher top-action agreement at least 95%;
- mean stored-vs-clean TV at most 0.05;
- stored target verified action-value delta vs raw has a positive 95% lower bound;
- clean teacher verified action-value delta vs raw has a positive 95% lower bound.

If the replay passes, one legal-masked learner transfer run uses the already frozen 200-step,
batch-64, LR 0.0002 and policy/value/calibration/tactical gates. No setting changes are permitted
after results. Failure blocks arena and returns the investigation to data scale, target balance,
or network capacity.
