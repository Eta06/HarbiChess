# UYUM common-direction policy target preregistration

Date: 2026-08-28  
Parent evidence: DENGE teacher gate failure  
Baseline and positions: unchanged MIHENK/KOPRU release checkpoint; exact frozen 96 train and 48 validation rows  
Decision scope: target qualification, then conditional fixed-compute learner ablation

## Hypothesis

The equal 512/800 DENGE mixture was globally stronger and stable but missed target coverage because it retained probability changes supported by only one search budget. UYUM tests a common-direction projection: change the raw-network policy only where both 512 and 800 independently agree on whether an action should gain or lose probability.

For raw probability `r_i` and clean-search probabilities `p512_i`, `p800_i`:

- common uplift is `u_i = min(max(p512_i - r_i, 0), max(p800_i - r_i, 0))`;
- common reduction is `d_i = min(max(r_i - p512_i, 0), max(r_i - p800_i, 0))`;
- subtract every `d_i` from raw;
- redistribute the total removed mass across all actions in proportion to `u_i`.

If no common uplift or reduction exists, the target remains raw. No action is selected by argmax, no action is pruned, and no temperature or learned coefficient is introduced. Multiple near-tied actions share mass whenever both searches support them.

The stability anchor applies the identical projection to raw plus 256/512 policies. Every legal action is again evaluated by the unchanged deterministic depth-4 verifier. Bootstrap uses 2,000 samples and seed `2026082819`.

## Frozen teacher gate

Use the exact DENGE row and aggregate metrics, definitions, and thresholds without relaxation:

- row anchor-to-target TV at most `0.20`;
- non-empty top-two overlap with the 256/512 projection;
- expected verified improvement over raw policy at least `+0.02`;
- verified harmful probability mass at most `0.10`;
- validation qualified ratio at least `20%`;
- harmful-row ratio at most `10%`;
- strictly positive bootstrap 95% lower bounds for all-row and qualified-row improvement;
- mean anchor-to-target TV at most `0.125`;
- mean verified expected value no more than `0.01` below the clean 800 policy.

Results cannot modify these gates. Failure blocks learner, arena, generation, and promotion.

## Conditional learner ablation

Run only after a teacher-gate pass. Compare the same three frozen arms and compute used in DENGE: `raw-control`, `single-800`, and `agreement-target`; exact 96/48 rows, policy-only AdamW, 240 steps, batch 64, learning rate `2e-4`, zero weight decay, identical game-balanced samples and checkpoint schedule.

Non-qualified rows use raw policy in every teacher arm. The unchanged full KOPRU validation replay measures WDL and calibration retention. UYUM must improve qualified-target legal cross-entropy by at least 2% from baseline, beat the single-800 arm, preserve target probability-mass capture, preserve raw and 64-search tactical solve counts, remain within `1.02x` baseline WDL cross-entropy and `+0.02` expected-score ECE, and keep finite gradients within the clipped norm limit of 5.0.

Passing this ablation does not authorize arena, generation, or promotion. It establishes only that a qualified search improvement can transfer into the current learner representation.
