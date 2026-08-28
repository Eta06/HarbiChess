# AKIS uncertainty-policy transfer preregistration (2026-08-28)

## Hypothesis

The qualified 512/800 uncertainty target transfers more faithfully when it
supervises the deployed policy logits directly. Close-valued actions remain a
soft distribution; zero-confidence branches contribute no target mass. The
frozen shared trunk and WDL head prevent a small policy experiment from hiding
value/calibration regression.

## Frozen evidence and compute

- Labels: `artifacts/diagnostics/bag-uncertainty-labels-20260828-01/labels.json`
- Verifier values: `artifacts/diagnostics/bag-raw-action-value-dataset-20260828-01/dataset.json`
- Baseline: KOPRU qualified replay baseline
- Train/validation positions: the already-frozen 96/48 fresh BAG split
- Trainable parameters: existing `policy_linear` only
- Optimizer: AdamW, learning rate `2e-4`, weight decay `0`
- Batch size: 16, seed `2026082833`, maximum 480 steps
- Checkpoints: 0, 60, 120, 240, 480; no early stopping

## Candidate gate

A nonzero checkpoint passes only if all conditions hold on the untouched BAG
validation split:

1. uncertainty-target legal-policy cross entropy improves by at least 5%;
2. predicted-policy/teacher Spearman is at least 0.35;
3. selected-action verified improvement over baseline has a positive 95%
   paired-bootstrap lower bound;
4. harmful selected-action ratio is at most 10%, mean verified regret at most
   0.10, and verified-best action coverage in policy top 16 at least 80%;
5. WDL logits remain bitwise identical;
6. raw-policy and 64/512-search tactical solved counts do not regress;
7. all gradients remain finite and within the existing safety policy.

Passing authorizes a separate frozen search-strength qualification only. It
does not authorize arena, generation, promotion, or any threshold change.
