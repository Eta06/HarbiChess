# BAGLANTI relational policy representation preregistration (2026-08-28)

## Hypothesis

YAPI failed because an origin-only spatial head cannot directly condition a
move on its destination. A shared relational scorer over origin trunk feature,
destination trunk feature, and move-plane embedding should fit the qualified
SIPER policy improvement with low parameter count and better action geometry.

## Frozen design

- Shared inputs per action: origin feature, destination feature, 8-wide plane
  embedding
- Hidden layer: 16 ReLU units; zero-initialized scalar output
- Invalid/off-board action encodings fall back to origin but remain masked
- Base policy, trunk, and WDL remain frozen
- VERI train split and SIPER target; no validation access
- AdamW `1e-3`, weight decay 0, batch 16, 480 steps
- Seed `2026082842`; no early stopping or projection

The unchanged YAPI gates apply: at least 20% reducible-gap closure, teacher
Spearman at least 0.35, positive verified-gain lower bound, harmful ratio at
most 10%, regret at most 0.10, top-16 coverage at least 80%, bitwise-stable WDL,
and finite gradients within norm 5.0.

Passing authorizes exactly one fresh 96-position teacher-validation set and
then a separate search qualification. It does not authorize arena, generation,
or promotion.
