# AKTAR Full Gumbel Learner-Transfer Preregistration

Date: 2026-08-28  
Status: frozen before target generation and training

## Question

Can the existing network absorb the qualified 256-simulation Full Gumbel soft
policy while retaining tactical ability, WDL calibration, and search strength?

This pilot isolates policy transfer. The trunk and complete WDL head are frozen
bitwise; only `policy_conv` and `policy_linear` may change. Continuous learning,
new self-play generation, and promotion remain disabled during the experiment.

## Frozen source and split

- Baseline model: KOPRU baseline SHA-256
  `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`.
- Train shard: KOPRU qualified replay `train-00000.jsonl.gz`.
- Validation shard: KOPRU qualified replay `validation-00000.jsonl.gz`.
- Deterministic selection seed: `2026082879`.
- Target rows: 384 train and 192 validation positions.
- Selection is game-balanced and stratified across opening/middlegame/endgame,
  tactical/quiet, known winning/drawing/losing, and unknown max-ply labels.
- Existing game-disjoint train/validation partitions are preserved.

## Frozen teacher and target

- Shared-tree Full Gumbel, 256 simulations, 16 considered root actions.
- Clean evaluation: Gumbel scale 0, no Dirichlet noise.
- Target is the normalized Mctx-style
  `softmax(raw_policy_logit + completed_q)` distribution, not root visit counts.
- Store source record identity, shard/model hashes, algorithm/config version,
  network prior, teacher target, selected action, root value, and root visits.
- Re-run 32 deterministically selected positions twice. Every selected action,
  visit count, root value, and target probability must reproduce within `1e-12`.
- Every target must be finite, normalized within `1e-9`, and supported only on
  legal actions. Any provenance or determinism failure blocks training.

## Frozen learner

- Trainable parameters: policy head only.
- Optimizer: Adam, learning rate `2e-4`, no weight decay.
- Batch size: 64, maximum 240 updates.
- Validation every 20 updates; select minimum validation teacher cross-entropy.
- Early stopping patience: four validation checks.
- No target exposure multiplier, top-action auxiliary loss, continuation target,
  replay recency weighting, or post-result hyperparameter adjustment.

## Transfer gates

All must pass:

1. Validation teacher cross-entropy improves by at least `0.01` absolute.
2. Validation teacher KL(candidate) improves by at least `0.01` absolute.
3. Validation teacher top-action agreement improves by at least two percentage
   points and does not trail train agreement by more than 15 points.
4. Candidate raw tactical solve count does not fall below baseline raw.
5. Candidate 256 Full Gumbel tactical solve count is at least 4/8 and loses no
   case solved by the baseline 256 Full Gumbel teacher.
6. Frozen non-policy parameters are byte-identical.
7. Validation WDL logits change by at most `1e-7`; WDL CE, Brier score, expected
   score correlation, and 10-bin ECE change by at most `1e-7`.

## Search-strength gate

Only a candidate passing gates 1-7 enters a fresh paired arena:

- Candidate 128 Full Gumbel versus baseline 128 Full Gumbel.
- 32 fresh opening pairs, color-swapped: 64 games.
- Opening seed `2026082893`, eight opening plies, max 256 plies.
- Candidate score at least 50%, paired 95% lower bound at least 45%, decisive
  score at least 50%, and max-ply/threefold rates no more than ten percentage
  points worse than the baseline side's corresponding rates.

## Authorization

- Passing every target, transfer, tactical, WDL, and search-strength gate
  authorizes implementation of rolling replay plus persistent latest-network
  policy iteration.
- It does not promote a release champion and does not authorize a large run.
- Any failure freezes this candidate and returns the audit to learner
  representation/optimization without changing these thresholds.
