# KOK joint representation transfer preregistration (2026-08-28)

## Hypothesis

CIPA showed that a rank-32 update to the frozen release policy representation
cannot transfer SIPER to unseen games even when broad replay drift is
constrained. The release trunk was originally learned from a weak early loop
and remains nearly uninformative for the current high-budget target. Updating
the trunk and policy head together may learn a transferable representation;
broad policy and WDL distillation must prevent unrelated behavior drift.

## Frozen experiment

- Target, verifier evidence, game split, and 2,048 replay anchors: identical to CIPA
- Trainable parameters: release trunk and policy head; the architecture is unchanged
- Steps: 960; target batch 16; anchor batch 64; AdamW `2e-4`, weight decay 0
- Target loss: SIPER soft-target cross entropy
- Replay losses: `KL(baseline policy || candidate policy)` and
  `KL(baseline WDL || candidate WDL)` on the same broad anchor states
- Policy-anchor arms: `1.0`, `4.0`, `16.0`
- WDL-anchor weight: `4.0` in every arm
- Identical sampler/optimizer seed for all arms: `2026082856`

The unchanged CIPA unseen-game gates apply: at least 20% reducible-gap closure,
teacher Spearman at least 0.35, positive verified-gain lower bound, harmful
ratio at most 10%, regret at most 0.10, top-16 coverage at least 80%, and
gradient norm at most 5.0. Broad replay policy KL must be at most 0.02. Broad
replay WDL KL must be at most 0.002 and mean absolute expected-score drift at
most 0.02.

Among passing arms choose the lowest holdout target cross entropy; exact ties
choose the larger policy-anchor weight. Failure emits no checkpoint. Passing
authorizes only an entirely fresh external teacher/target, tactical, WDL,
calibration, and search-strength qualification. It does not authorize arena,
generation, or promotion.
