# FREN policy-delta projection preregistration (2026-08-28)

## Hypothesis

SIPER's target is qualified and rank-32/`1e-3` absorbs it, but the unscaled
policy update crosses argmax boundaries too aggressively. A single global
train-only projection of the learned delta can preserve verified gain while
keeping harmful action selection inside the frozen safety limit.

## Frozen fit and projection

- Source: SIPER target and VERI train partition only
- Adapter: rank 32, learning rate `1e-3`, batch 16, 480 steps
- Seed: `2026082840`; weight decay 0; no early stopping
- Projection grid: `0.1, 0.2, ..., 1.0` times the merged policy-weight delta
- Projection selection: largest scale satisfying every train safety gate
- No VERI validation metric participates in scale selection

A scale qualifies only when:

1. it closes at least 20% of the mathematically reducible target KL gap,
   computed as `(baseline CE - projected CE) / (baseline CE - target entropy)`;
2. teacher-policy Spearman is at least 0.35;
3. verified selected-action gain has a positive 95% lower bound;
4. harmful selected-action ratio is at most 10%;
5. mean verified regret is at most 0.10;
6. best-action top-16 coverage is at least 80%;
7. gradients are finite and within norm 5.0.

If no scale passes, representation redesign is required. If a scale passes,
the merged checkpoint may be evaluated once on a completely fresh 96-position
teacher-validation set. That later validation must independently satisfy the
same behavior gates, at least 20% reducible-gap closure, bitwise-identical WDL,
and non-regressing raw/64/512 tactical solve counts. Neither stage authorizes
arena, generation, or promotion; passing fresh validation authorizes only a
separate search-strength qualification.
