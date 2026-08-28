# SIPER conservative policy-target preregistration (2026-08-28)

SIPER freezes every KILAVUZ rule and guardrail. The only change is the Q
statistic inside the KL-constrained mirror-descent update: each qualified
action uses `min(Q512, Q800)` instead of the visit-weighted cross-budget mean.
This tests whether a conservative value estimate removes harmful target leaders
without sacrificing the already-positive expected-value distribution.

The same VERI 379/95 labelled split, raw policy, KL cap 0.10, bootstrap size,
and all target gates are reused. In particular, target-top harmful ratio must
remain at most 10%; expected-value gain must retain a positive 95% lower bound;
effective-action ratio must remain at least 50%; and no thresholds may change
after results.

Only a passed target may enter the unchanged AKIS learner ablation: rank 8,
480 steps, batch 16, AdamW `2e-4`, checkpoints 0/60/120/240/480, fresh learner
seed `2026082838`, and the same imitation, verifier, WDL, and tactical gates.
Passing learner evidence authorizes only a separate search qualification, not
arena, generation, or promotion.
