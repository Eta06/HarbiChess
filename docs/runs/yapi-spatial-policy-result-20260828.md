# YAPI spatial policy representation result (2026-08-28)

## Decision

The origin/plane spatial adapter failed its train-fit gate. Fresh validation,
search qualification, arena, generation, and promotion remain blocked.

## Evidence

After the frozen 480 steps, the 1,241-parameter adapter closed only 6.27% of
the reducible KL gap. Teacher-policy Spearman was 0.2789 and harmful selected
actions reached 15.30%. The verified-gain lower bound was positive and WDL
logits were bitwise unchanged, but three mandatory gates failed.

An origin-square `1×1` policy plane does not expose the destination feature
needed to distinguish many chess moves. The next representation hypothesis is
a shared relational adapter using origin trunk features, destination trunk
features, and move-plane embeddings under the same compute and safety gates.

## Frozen artifact

- `artifacts/diagnostics/yapi-spatial-policy-20260828-01/result.json`
