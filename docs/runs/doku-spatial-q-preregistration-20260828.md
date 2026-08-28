# DOKU spatial action-value transfer preregistration

Date: 2026-08-28  
Parent: DEGER global-linear action-value transfer failure  
Decision scope: fresh Q-label qualification and frozen spatial-head transfer; no generation, arena, or promotion

## Hypothesis

DEGER preserved policy/WDL behavior exactly but failed validation because its global `256 → 4,672` action matrix added roughly 1.2 million trainable parameters for only 96 labelled positions. HarbiChess's action schema is already spatial: every index is one of 73 move planes attached to one of 64 canonical origin squares.

DOKU replaces the global action matrix with a shared `1×1 convolution` from the frozen trunk to 73 advantage planes. Flattening the `8×8×73` output exactly matches the existing square-major policy action encoding. The head retains the dueling `Q(s,a) = tanh(V(s) + A(s,a))` form and zero initialization, but has about 4–5 thousand trainable parameters and shares move-pattern learning across all squares.

## Fresh non-overlapping labels

From the existing qualified KOPRU replay, select 96 train and 48 validation positions with stratification seed `2026082824`, excluding every game/index/ply identity used by MIHENK/TERAZI/DEGER. Train and validation games remain disjoint.

Run clean 512/800 PUCT with the unchanged release model, depth-1 oracle, 24 root workers, eight oracle workers, batch cap 48, and wait 0.25 ms. Evaluate every legal action with the deterministic depth-4 verifier. The training Q label and visit-confidence weight use exactly the DEGER formula: visit-count-weighted 512/800 Q mean on common visited support and square-root minimum-visit weight normalized per position.

Before training, fresh labels must satisfy:

- mean 800-Q versus verifier Spearman at least `0.35`;
- mean 512/800 Q Spearman at least `0.70`;
- mean absolute cross-budget Q drift at most `0.03`;
- mean top-two Q-set overlap at least `75%`;
- 800 top-Q verified delta has a strictly positive bootstrap 95% lower bound;
- 800 top-Q harmful ratio at most `10%` and mean regret at most `0.10`.

This set-level gate intentionally represents near-tied uncertainty and does not reinterpret TERAZI's failed top-one gate as a pass. Failure blocks training.

## Frozen transfer

If the fresh label gate passes, train only the spatial action-value head:

- 73 output planes directly; no hidden action-value channels or global linear layer;
- AdamW `2e-4`, zero weight decay;
- batch 16, exactly 480 steps, no early stopping;
- gradient clipping 5.0, sampler seed `2026082825`;
- checkpoints `0, 60, 120, 240, 480`.

Reuse every DEGER transfer threshold unchanged: at least 20% validation Q-MSE improvement, teacher-Q Spearman at least 0.35, positive verified top-action interval, harmful ratio at most 10%, regret at most 0.10, top-16 best-action coverage at least 80%, policy/WDL logit delta at most `1e-7`, exact tactical retention, and finite gradients within the clipping limit. Bootstrap uses 2,000 samples and seed `2026082826`.

No thresholds, duration, architecture width, or position count may change after results are visible. A pass authorizes only a separately preregistered completed-Q search qualification.
