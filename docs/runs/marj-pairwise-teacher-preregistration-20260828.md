# MARJ uncertainty-aware teacher preregistration (2026-08-28)

## Hypothesis

Per-row Spearman is mis-specified for flat chess positions because it penalizes
arbitrary ordering among near-equal actions. Teacher ordering should be judged
only where an independent verifier finds a meaningful action-value margin;
near ties remain in the soft target and are not converted into hard labels.

## Frozen diagnostic metric

- Source diagnostic: SINAV fresh teacher rows (never promotion evidence)
- Teacher score: conservative `min(Q512, Q800)` on drift-qualified actions
- Informative pair: absolute verifier-value difference at least `0.05`
- Pair score: 1 for matching order, 0 for reversed order, 0.5 for Q tie
- Row score: mean over informative pairs; dataset score: mean over rows
- Bootstrap: 2,000 row-level samples, seed `2026082846`

The uncertainty-aware teacher passes only if:

1. at least 50% of labelable positions contain an informative pair;
2. mean decisive-pair concordance is at least 0.60 and its 95% lower bound is
   above 0.50;
3. labelable ratio is at least 95%, common support at least 95%, and stable
   visit mass at least 80%;
4. conservative verified-gain lower bound remains positive;
5. conservative harmful ratio is at most 10% and regret at most 0.10.

This replaces only the flat-position Spearman gate; no strength/safety threshold
is relaxed. If SINAV supports the metric, the identical gate must pass on a new
fresh 96-position set before the YAKINSAMA checkpoint may be evaluated. No
stage authorizes arena, generation, or promotion.
