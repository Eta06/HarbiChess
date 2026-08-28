# AKTAR Policy-Anchor Transfer Preregistration

Date: 2026-08-28  
Status: frozen before training

## Hypothesis

The unanchored policy head learned the qualified teacher but changed interior
Full Gumbel allocation enough to amplify weak leaf-Q noise and lose both forced
defenses. A baseline-policy trust region may preserve tactical search while
retaining measurable teacher imitation.

## Frozen ablation

- Every arm starts from the original KOPRU baseline, never the rejected
  unanchored candidate.
- Same 384/192 qualified target rows, policy-only representation, Adam `2e-4`,
  batch 64, maximum 240 steps, validation every 20, patience four.
- Frozen trunk and WDL head remain byte-identical.
- Training target is `(teacher + anchor_weight * baseline_policy) /
  (1 + anchor_weight)`.
- Anchor weights: `0.5`, `1.0`, `2.0`, `4.0`.
- No target/search/exposure/training-duration changes.

## Arm gates

Each arm must pass all original pre-arena gates:

1. teacher validation CE and KL improve by at least 0.01;
2. teacher top-action agreement improves by at least two points and validation
   trails train by no more than 15 points;
3. raw tactical solve count does not regress;
4. 256 Full Gumbel retains at least 4/8 and every baseline-solved case;
5. WDL metrics/logits and frozen parameter hash remain unchanged;
6. mean validation `KL(baseline || candidate)` is at most 0.10.

Among eligible arms select minimum original-teacher validation CE, breaking ties
toward the larger anchor. Open exactly one fresh paired arena for that selected
arm using the already frozen arena seed, openings, budgets, and gates. If no arm
passes, continuous learning remains blocked and the next investigation moves to
value/representation rather than weakening these gates.
