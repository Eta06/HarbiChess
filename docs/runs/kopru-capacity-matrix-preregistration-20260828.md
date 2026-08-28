# KOPRU capacity and representation matrix preregistration

Date: 2026-08-28  
Replay: `kopru-qualified-replay-20260828-01`  
Teacher alignment: `kopru-qualified-replay-alignment-20260828-01`  
Promotion state: arena, learner continuation, and new generation remain disabled

## Hypothesis

The qualified clean search teacher improves verified action value in aggregate, but the current `16 trunk channels / 2 residual blocks / 4 policy channels` learner reduces soft policy cross-entropy while losing teacher top-action agreement on both train and validation replay. A small fixed subset can be overfit, whereas one to four full-replay epochs do not recover validation top-action agreement. This matrix tests whether shared trunk depth or policy-head representation capacity is the limiting factor without changing replay, search targets, exposure, optimizer, or sampling.

## Frozen matrix

| ID | Trunk channels | Residual blocks | Policy channels | Value channels | Value hidden |
|---|---:|---:|---:|---:|---:|
| `base` | 16 | 2 | 4 | 2 | 32 |
| `deep` | 16 | 4 | 4 | 2 | 32 |
| `head` | 16 | 2 | 8 | 2 | 32 |
| `deep-head` | 16 | 4 | 8 | 2 | 32 |

Expanded models must be initialized as function-preserving copies of the release baseline. Added residual blocks start as identity blocks. Duplicated policy channels divide their outgoing linear weights so that initial policy and WDL logits match the baseline. An expansion is invalid if the maximum initial logit difference on the validation set exceeds `1e-5`.

## Frozen learner conditions

- Policy-only learner; value-head weight is zero.
- AdamW, learning rate `2e-4`, weight decay zero, gradient limit `5.0`.
- Batch size `64`.
- Game-balanced sampling seed `2026082814`.
- Exactly two replay-equivalent epochs: `ceil(15122 / 64) * 2 = 474` steps.
- No continuation/repetition adjustment and no hard-top auxiliary.
- Metrics are recorded before training and at `0.5`, `1.0`, and `2.0` epochs.
- The same record order, sampler seed, and evaluation set are used for every architecture.

## Frozen decision gate

The final two-epoch checkpoint is the decision point; intermediate checkpoints cannot change the sample size, duration, or thresholds. An expanded architecture is a useful transfer candidate only if all conditions hold:

1. Validation legal teacher-policy cross-entropy is no worse than the two-epoch `base` control.
2. Validation teacher top-action agreement exceeds the untrained release baseline (`40.7067%`) and improves on the trained `base` control by at least `3.0` percentage points.
3. Raw-policy and 64-simulation tactical solve counts do not regress from the release baseline.
4. Losses and gradients remain finite and the gradient norm stays within the existing safety limit.

Inference latency and throughput are measured at masked batches `4`, `16`, and `64`. They are reported as a deployment tradeoff, not used to rescue a failed learning gate. Passing this diagnostic does not authorize arena, promotion, or a new generation; it only identifies the architecture for the next separately preregistered learner-transfer confirmation.

## Performance profiling boundary

Capacity training is timed as replay read, reconstruction/encoding, MLX preparation, optimizer steps, validation evaluation, and tactical evaluation. Search profiling continues independently with the existing baseline checkpoint and frozen search workload so architecture changes cannot confound CPU/tree/batching measurements. Only changes that improve end-to-end wall-clock throughput while preserving exact rules and search outputs are retained.
