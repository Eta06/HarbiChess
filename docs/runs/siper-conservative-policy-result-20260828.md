# SIPER conservative policy result (2026-08-28)

## Decision

The SIPER target passed, but its frozen learner transfer failed. No search
qualification, arena, generation, or promotion is authorized.

## Target

Using `min(Q512, Q800)` retained positive validation expected-value gain
(95% interval +0.00920 to +0.01677), 90.50% effective-action ratio, 8.42%
harmful target-top actions, zero harmful expected-value rows, and 0.0551 mean
target-top regret. It passed every unchanged KILAVUZ guardrail.

## Learner

The rank-8, 480-step adapter preserved WDL logits exactly. At steps 240 and
480 its validation selected-action gain intervals were positive, but no
checkpoint passed all gates. Validation teacher Spearman remained at or below
0.1171, cross entropy never improved by 5%, harmful selection was 11.58%, and
mean regret remained above 0.10. The maximum gradient norm was only 0.0535.

The target is independently qualified and produces a real verified-gain
signal, but the frozen adapter optimization underfits it. Rank and optimizer
step size must be diagnosed on the train partition only; any selected transfer
configuration then requires a fresh validation teacher set.

## Frozen artifacts

- `artifacts/diagnostics/siper-policy-target-20260828-01/targets.json`
- `artifacts/runs/siper-policy-transfer-20260828-01/result.json`
