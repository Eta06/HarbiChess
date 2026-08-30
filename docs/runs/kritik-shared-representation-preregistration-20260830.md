# KRITIK shared-representation joint-transfer preregistration

## Hypothesis

The frozen release trunk was learned primarily through the policy path and exposes insufficient
features to the WDL head. Head-only training therefore memorizes within-game positions but cannot
generalize outcome value across games. The correct controlled test is to train the shared trunk,
policy head, and value head jointly against the already-qualified Full Gumbel policy target and
corrected terminal WDL target.

The failed head-only warmup remains evidence and its gate is not relaxed. For this representation
audit only, it no longer prevents the separately identified all-network joint arm from running.

## Frozen inputs and schedule

- Baseline SHA-256: `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`.
- Full Gumbel target: `aktar-full-gumbel-targets-20260828-05`, exactly 384 train and
  192 validation positions.
- Value replay: schema-12 `kopru-qualified-replay-20260828-01`; unknown/max-ply rows excluded.
- Shared network starts from the release baseline, not from the failed value-head warmup.
- Separate policy and outcome/game-balanced value batches, each size `64`.
- AdamW, learning rate `1e-4`, weight decay `1e-4`, maximum `400` steps.
- Validation every `20` steps, patience `6`, seed `2026083017`.
- Search allocator, teacher targets, exposure, architecture, and gate thresholds stay unchanged.

## Frozen pre-arena gates

Every selected checkpoint must simultaneously satisfy:

- validation micro and macro WDL CE improvement at least `0.10`;
- validation Brier improvement at least `0.03`;
- validation expected-score Pearson at least `0.20`;
- ordered loss/draw/win means with both adjacent margins at least `0.03`;
- validation ECE-10 at most `0.12` and train/validation macro-CE gap at most `0.15`;
- Full Gumbel policy validation CE improvement at least `0.05`;
- Full Gumbel top-action agreement gain at least `2` percentage points;
- policy validation agreement trails train by no more than `15` points.

Only a checkpoint passing all numeric gates may run continuation action-value ranking and the
existing Full Gumbel tactical suite. Arena runs only if continuation ranking improves and the
candidate keeps at least `4/8` Full Gumbel tactical solves without losing a baseline-solved case.

## Decision rule

- Passing all gates supports the shared-representation hypothesis and permits design of the
  continuous learner, but does not authorize generation or promotion.
- Policy transfer without WDL qualification, or train-only WDL improvement, is failure. Audit
  value targets, representation/head, and loss structure next; do not lengthen training.
- WDL qualification with tactical/policy regression is also failure and requires a policy-
  preservation or decoupled representation experiment before arena.

