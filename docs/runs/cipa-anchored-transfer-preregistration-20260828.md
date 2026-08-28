# CIPA replay-anchored learner preregistration (2026-08-28)

## Hypothesis

KANIT proved that sparse high-budget target fitting can improve train behavior
while damaging unseen policy states. A broad replay distillation anchor should
let the learner absorb SIPER improvements while constraining off-target policy
drift.

## Frozen internal experiment

- Source high-budget data: 379 VERI/SIPER train rows
- Split unit: whole game, preventing positions from one game crossing sides
- Holdout games: deterministic 20% by SHA-256 order, seed `2026082852`
- Anchor states: 2,048 unique positions from fit-side replay games
- Adapter: rank 32; 960 steps; high-target batch 16; anchor batch 64
- Optimizer: AdamW `1e-3`, weight decay 0
- Anchor loss: `KL(baseline policy || candidate policy)` on legal moves
- Arms: anchor weights `0.25`, `1.0`, `4.0`
- Arm seeds: `2026082853`, `2026082854`, `2026082855`

An arm passes only if the unseen-game internal holdout closes at least 20% of
its reducible target gap, teacher Spearman is at least 0.35, verified-gain lower
bound is positive, harmful ratio at most 10%, regret at most 0.10, and top-16
coverage at least 80%. Mean anchor KL must be at most 0.02 and gradients finite
within norm 5.0. Among passing arms choose lowest holdout CE; exact ties choose
the larger anchor weight.

If no arm passes, replay anchoring is rejected. A selected arm must still pass
an entirely new external teacher/target validation and tactical gate. Nothing
in CIPA authorizes search qualification, arena, generation, or promotion.
