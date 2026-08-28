# AKTAR Full Gumbel Learner-Transfer Result

Date: 2026-08-28  
Artifact: `artifacts/runs/aktar-full-gumbel-transfer-20260828-01/result.json`

## Decision

The frozen policy-head learner transfer failed its tactical retention gate.
Arena, continuous learning, generation, and promotion remain blocked. The
candidate is rejected and will not be modified.

## Successful transfer signals

- Selected checkpoint: step 40; early stopping completed at step 120.
- Validation teacher CE: 2.87061 -> 2.76729 (improvement 0.10332).
- Validation teacher KL: 1.84331 -> 1.73999 (improvement 0.10332).
- Validation teacher top-action agreement: 8.85% -> 15.63%.
- Train top-action agreement: 13.54% -> 25.52%; validation trails by 9.90
  points, within the frozen 15-point limit.
- Raw tactical solve count: 1/8 -> 2/8.
- Maximum WDL-logit delta: exactly 0.0.
- WDL CE, Brier, correlation, and ECE: exactly unchanged.
- Frozen trunk/value parameter hash: exactly unchanged.

## Blocking regression

- Baseline 256 Full Gumbel tactical: 4/8.
- Candidate 256 Full Gumbel tactical: 2/8.
- Both forced-defense cases were lost; only the two mate-in-one cases remained.

The regression is not root-action prior imitation failure: candidate raw policy
correctly chose both forced defenses and increased their expected policy mass.
The policy update changed interior Full Gumbel allocation enough that, after 128
visits to each of the two root actions, tiny backed-up Q differences selected the
losing defense. This localizes the next learner problem to unconstrained policy
drift interacting with weak/noisy leaf value, rather than inability to reduce
teacher imitation loss.

## Next hypothesis

Keep the qualified targets, compute, frozen trunk, and WDL head. Test a
preregistered policy-distillation trust region that anchors the candidate to the
baseline legal policy while learning the Full Gumbel target. It must retain the
same CE/KL/top-action gains and restore all baseline-solved tactical cases before
one fresh arena is opened. Do not tune search, target exposure, or training
length from this result.
