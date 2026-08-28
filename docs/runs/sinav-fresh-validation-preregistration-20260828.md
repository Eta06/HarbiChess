# SINAV fresh policy-transfer validation preregistration (2026-08-28)

## Frozen candidate and data

- Candidate: YAKINSAMA step 960, SHA-256
  `c114b9cffce0f12c993f572de2641fedab012a027f4792bea3030b9054c33e2a`
- Baseline: unchanged KOPRU baseline
- Validation: 96 fresh positions from the existing qualified replay
- Exclusions: every TERAZI, DOKU, OLCEK, BAG, and VERI diagnostic identity
- Clean search: 512/800; dataset seed `2026082844`
- Uncertainty bootstrap seed: `2026082845`
- Target: conservative-Q SIPER rule, KL cap 0.10
- No candidate retraining, scaling, checkpoint selection, or threshold change

The fresh target must first pass its existing uncertainty and policy-target
gates. The frozen candidate then passes transfer only if it closes at least 20%
of the fresh reducible KL gap, teacher Spearman is at least 0.35, verified-gain
95% lower bound is positive, harmful selection is at most 10%, regret at most
0.10, top-16 coverage at least 80%, WDL logits are bitwise identical, and
raw/64/512 tactical solved counts do not regress.

Passing authorizes only a separate candidate-vs-baseline search-teacher
qualification on fresh positions. It does not authorize arena, generation, or
promotion.
