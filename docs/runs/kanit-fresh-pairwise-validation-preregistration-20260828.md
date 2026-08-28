# KANIT fresh pairwise teacher validation preregistration (2026-08-28)

## Frozen evidence chain

- Candidate: YAKINSAMA step 960, SHA-256
  `c114b9cffce0f12c993f572de2641fedab012a027f4792bea3030b9054c33e2a`
- Fresh validation positions: 96
- Exclusions: TERAZI, DOKU, OLCEK, BAG, VERI, and SINAV identities
- Dataset seed: `2026082847`
- Uncertainty bootstrap seed: `2026082848`
- Decisive-pair bootstrap seed: `2026082849`
- Policy-target bootstrap seed: `2026082850`
- Search budgets: clean 512/800; verifier depth 4

The unchanged MARJ pairwise teacher gate runs first. Only if it passes may the
unchanged conservative-Q SIPER target be materialized and gated. Only if both
pass may the frozen candidate be evaluated once with the SINAV transfer gates:
20% reducible-gap closure, teacher Spearman at least 0.35, positive verified
gain lower bound, harmful ratio at most 10%, regret at most 0.10, top-16
coverage at least 80%, bitwise-stable WDL, and non-regressing raw/64/512
tactical solved counts.

No retraining, scaling, checkpoint change, or threshold adjustment is allowed.
Passing candidate validation authorizes only a separate fresh search-strength
qualification. It does not authorize arena, generation, or promotion.
