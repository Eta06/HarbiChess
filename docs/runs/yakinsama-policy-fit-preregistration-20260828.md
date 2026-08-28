# YAKINSAMA low-rank policy convergence preregistration (2026-08-28)

## Hypothesis

The qualified SIPER target has 7.39% harmful top actions on train, while the
rank-32 adapter at 480 steps remains at 11.35% harmful and only 46% of the
reducible KL gap closed. If incomplete optimization is the remaining blocker,
continued fixed-configuration fitting should approach the safer target rather
than require a new heuristic or larger representation.

## Frozen diagnostic

- Data: VERI train partition and SIPER target only
- Adapter: rank 32; AdamW `1e-3`; weight decay 0
- Batch: 16; seed `2026082843`
- Checkpoints: 480, 960, 1920; fixed maximum 1920
- No validation access, projection, early stopping, or threshold change

At each checkpoint the existing behavior gates apply using the mathematically
valid gap metric: reducible-gap closure at least 20%, teacher Spearman at least
0.35, positive verified-gain lower bound, harmful ratio at most 10%, regret at
most 0.10, top-16 coverage at least 80%, and finite gradient norm at most 5.0.
The earliest passing checkpoint is selected without looking at validation.

If none passes, longer fitting is rejected as a remedy. If one passes, its
merged checkpoint may be evaluated once on a completely fresh 96-position
teacher-validation set. This stage does not authorize search qualification,
arena, generation, or promotion by itself.
