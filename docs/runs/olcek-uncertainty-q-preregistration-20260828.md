# OLCEK uncertainty-weighted spatial-Q preregistration

Date: 2026-08-28  
Parent: DOKU label gate failure and action-level Q-drift audit  
Decision scope: fresh uncertainty-aware label qualification and spatial-head transfer only

## Hypothesis

DOKU showed healthy full-Q correlation and verified strength, while a fixed top-two set failed because leader margins are about 0.01. Meanwhile 91.71% of visit-weighted action support had 512/800 Q drift at most 0.03. Treating all labels as equally certain discards this structure.

OLCEK stores the 512/800 absolute Q difference for every commonly visited action. Its target remains their visit-count-weighted Q mean. The loss weight is:

`sqrt(min(visits512, visits800)) * max(0, 1 - abs(Q512 - Q800) / 0.03)`

and is normalized within each position. Actions with disagreement at or above 0.03 remain in telemetry but receive zero supervised weight. This is continuous uncertainty weighting below the cutoff, not top-action pruning or a single-label target.

## Fresh data and teacher gate

Select a third exact set of 96 train and 48 validation positions using seed `2026082827`, excluding all identities from both TERAZI and DOKU. Run the unchanged clean 512/800 depth-1-oracle search and full legal-action depth-4 verifier with 24 root workers, eight oracle workers, batch cap 48, and wait 0.25 ms.

Before learner training, validation must satisfy:

- common visited support covers at least `95%` of legal actions;
- at least `80%` of minimum-visit-weighted common support has Q drift at most 0.03;
- mean Q/verifier Spearman on the drift-qualified support is at least `0.35`;
- selecting the maximum conservative `min(Q512,Q800)` has a strictly positive verified-delta bootstrap 95% lower bound;
- conservative selection harmful ratio is at most `10%` and mean verified regret at most `0.10`.

Bootstrap uses 2,000 samples and seed `2026082828`. Failure blocks training.

## Frozen spatial transfer

If labels pass, initialize a zero-advantage spatial `1×1 trunk → 73 action planes` head over the unchanged release network. Train only this head with AdamW `2e-4`, zero weight decay, batch 16, exactly 480 steps, clipping 5.0, sampler seed `2026082829`, and checkpoints `0, 60, 120, 240, 480`.

The DEGER transfer gates remain unchanged: at least 20% validation weighted-MSE improvement, teacher-Q Spearman at least 0.35 on supervised actions, positive depth-4 verified top-action interval, harmful ratio at most 10%, regret at most 0.10, predicted-Q top-16 best-action coverage at least 80%, policy/WDL logit delta at most `1e-7`, exact raw/64-search tactical retention, and finite gradients within clipping.

Checkpoint selection uses the complete gate and then lowest weighted Q MSE. No duration, threshold, drift cutoff, sample count, or head shape can change after results become visible. A pass authorizes only completed-Q search qualification; generation, arena, and promotion remain blocked.
