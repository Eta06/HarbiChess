# TESHIS teacher-instability segmentation (2026-08-28)

## Finding

SINAV's failure is concentrated in flat-value positions, not in general
cross-budget search instability. Overall cross-budget Q Spearman was 0.7615 and
stable visit mass was 88.70%, while stable-Q/verifier Spearman was 0.3152.

Worst segments were balanced material (rho 0.1651), losing value states
(0.1258), openings (0.2340), low branching (0.2097), and tactical positions
(0.2920). Winning states reached 0.5096. Low-branching and losing states also
had very small stable-Q spreads, 0.0414 and 0.0620 respectively.

Per-row Spearman treats every ordering swap as equally wrong even when action
values are effectively tied. The next diagnostic must keep those actions soft
and measure ordering only on verifier-separated action pairs, while retaining
the unchanged expected-value, harm, regret, coverage, and labelability gates.

## Frozen artifact

- `artifacts/diagnostics/teshis-teacher-instability-20260828-01/audit.json`
