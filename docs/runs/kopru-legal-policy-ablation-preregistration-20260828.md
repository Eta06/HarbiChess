# KOPRU legal-policy ablation preregistration

The first learner transfer run failed and remains rejected. Its global policy cross-entropy fell
from 8.4506 to 6.4243 while legal top-action agreement did not improve (11.13% to 10.80%) and
raw tactical solve count regressed from 1/8 to 0/8. This isolates an objective mismatch: search
and inference normalize over legal actions, while the learner normalized over all 4,672 actions.

This ablation changes exactly one training behavior: policy cross-entropy is normalized over the
full legal-action mask for each position. It also reports global cross-entropy for diagnosis, but
the preregistered teacher-imitation gate uses legal-masked cross-entropy. Applied clipped gradient
norm and pre-clip norm are recorded separately; only the applied norm is a safety gate.

Everything else remains frozen:

- Run ID: `kopru-learner-legal-mask-20260828-01`
- Baseline, train/validation replay, teacher audit, seed, LR, batch size, 200-step maximum,
  validation interval, patience, checkpoint count, and tactical budgets are unchanged.
- Required legal-masked teacher-policy CE improvement: at least 2%.
- WDL validation CE: no worse than 2% over baseline.
- Expected-score ECE: no worse than baseline +0.02.
- Raw tactical and 64/512 teacher-search tactical solve counts: no regression.
- No arena or promotion unless one checkpoint passes every gate.

No fresh self-play, continuation data, target heuristic, exposure increase, or network-capacity
change is permitted in this ablation. If legal masking improves imitation but still fails tactical
retention, the next diagnosis will test tactical replay scarcity/weighting or model capacity with
another separately preregistered controlled experiment.
