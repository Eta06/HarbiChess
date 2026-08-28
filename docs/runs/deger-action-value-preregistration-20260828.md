# DEGER action-value representation preregistration

Date: 2026-08-28  
Parents: TERAZI Q reliability and ODAK root-allocation qualification  
Decision scope: frozen offline representation transfer; no self-play generation, arena, or promotion

## Hypothesis

TERAZI proved that high-budget child-Q ranks independent verified action value better than visit count, while ODAK proved that concentrating root compute improves 512/800 decision stability but loses strength when candidates come only from the policy top 16. The current network has no state-action value representation: it must compress all search improvement into one global policy distribution and one scalar WDL value.

DEGER adds a dueling legal-action value head. It predicts `Q(s,a) = tanh(V(s) + A(s,a))`, where `V(s)` is the unchanged WDL expected value and `A` is a new convolution plus linear action-advantage head. The final advantage layer is initialized to exactly zero. Existing policy/WDL outputs and checkpoint behavior must therefore remain unchanged.

Only the new action-value head is trainable in this experiment. Trunk features and WDL state values are precomputed and detached, so no gradient or optimizer update can alter the release policy, trunk, or WDL parameters.

## Frozen labels and data

Use the exact TERAZI 96 train and 48 validation positions. For an action visited at both 512 and 800 simulations, its label is the visit-count-weighted mean of the two root child-Q estimates. Its loss weight is the square root of the smaller visit count, normalized within that position. Actions absent from either budget receive zero loss weight; an unvisited action is never assigned Q=0.

The independent depth-4 legal-action verifier is rerun only for evaluation and never used as a training label. This preserves the distinction between teacher imitation and external strength verification.

Frozen head/training configuration:

- action-value channels: 4;
- AdamW learning rate: `2e-4`, weight decay: zero;
- batch size: 16 positions;
- exactly 480 steps, no early stopping;
- gradient clipping: 5.0;
- sampler seed: `2026082822`;
- checkpoints: steps `0, 60, 120, 240, 480`.

The baseline control is the same zero-advantage head before training. Checkpoint selection evaluates the complete gate at every frozen checkpoint; validation MSE alone is not sufficient.

## Frozen transfer gate

At least one non-zero checkpoint must satisfy all conditions:

- validation weighted Q MSE improves by at least `20%` from the zero-advantage baseline;
- mean per-position Spearman correlation with teacher Q is at least `0.35`;
- the predicted-Q top action has a strictly positive bootstrap 95% lower bound for depth-4 verified delta versus raw-policy argmax;
- predicted-Q top-action harmful ratio is at most `10%` for delta at or below `-0.025`;
- mean predicted-Q top-action verified regret is at most `0.10`;
- predicted-Q top-16 contains an independently best legal action in at least `80%` of validation rows;
- maximum absolute change in release policy and WDL logits is at most `1e-7` on all validation rows;
- raw-policy and 64-search tactical solve counts exactly equal the release baseline;
- losses and gradients remain finite and clipped gradient norm does not exceed 5.0.

Bootstrap uses 2,000 samples and seed `2026082823`. Thresholds, checkpoints, duration, and selection rules cannot change after metrics become visible.

A pass proves offline learner transfer into the new representation and authorizes a separately preregistered completed-Q search qualification. It does not authorize large replay, arena, generation, or promotion. A failure rejects this head/loss contract; training duration, width, or exposure cannot be tuned on the same rows.
