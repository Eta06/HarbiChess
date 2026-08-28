# YAPI spatial policy representation preregistration (2026-08-28)

## Hypothesis

The transfer blocker is the global, weakly structured 4672-action policy
projection. A shared spatial residual head aligned with the existing
`origin-square × 73 move-plane` action encoding should absorb SIPER targets
with far fewer parameters and better positional generalization.

## Frozen representation and fit

- Input: frozen baseline trunk tensor
- Adapter: zero-initialized `1×1 Conv2d(trunk_channels, 73)`
- Output order: `(row, column, move_plane)` flattened to 4672 logits
- Trainable parameters: 1,241 at the current 16-channel trunk
- Base policy, trunk, and WDL parameters remain frozen
- Source: SIPER target and VERI train partition only
- Optimizer: AdamW `1e-3`, weight decay 0, batch 16
- Steps: 480; seed `2026082841`; no early stopping

Train-fit passes only with at least 20% reducible-KL-gap closure, teacher
Spearman at least 0.35, positive verified-gain 95% lower bound, harmful ratio
at most 10%, regret at most 0.10, top-16 best-action coverage at least 80%, and
finite gradients within norm 5.0.

Passing authorizes one evaluation on a completely fresh 96-position
teacher-validation set. That set must independently pass the same transfer
gates, preserve WDL logits bitwise, and not regress raw/64/512 tactical solve
counts. Only then may a separate candidate search-teacher qualification run.
No stage here authorizes arena, generation, or promotion.
