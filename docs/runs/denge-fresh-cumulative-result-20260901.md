# DENGE fresh cumulative pilot result

Run: `denge-continuous-pilot-20260901-01`

## Decision

**Failed.** Production continuous generation and promotion remain disabled.
All three rolling updates were locally accepted and the cumulative old/fresh
value gate passed every endpoint, but the preregistered continuation rule
allowed at most one verified-top loss. The final candidate lost 16 net hits
across 1,440 positions, so the run cannot be reclassified.

## Main evidence

- Update learner steps: 20, 60, 87; all exact-resume checks had zero parameter
  and metric delta.
- Fresh qualification: 968 known games. CE `0.72520 -> 0.61568`, Brier
  `0.44135 -> 0.36841`, ECE `0.11454 -> 0.05984`, Pearson
  `0.72150 -> 0.74852`; every paired confidence condition passed.
- Old qualification: 529 known games. CE `0.73116 -> 0.63035`, Brier
  `0.44296 -> 0.37884`, ECE `0.11636 -> 0.03597`, Pearson
  `0.67341 -> 0.69427`; every non-inferiority condition passed.
- Final arena: 64 games, 8 wins / 49 draws / 7 losses, score `0.50781`, paired
  interval `[0.45313, 0.55469]`, no threefold repetitions.
- Continuation Spearman delta: `-0.00539`, interval
  `[-0.01434, 0.00356]`; this endpoint passed its `-0.020` margin.
- Verified-top agreement: `0.52778 -> 0.51667`, 51 losses and 35 gains, net
  `-16/1440` (`-0.01111`). The legacy one-position limit failed.

## Audit finding

The 1,440 continuation positions came from only 48 historical game clusters.
The old top-action rule was neither a practical non-inferiority margin nor a
game-clustered statistical test. The replacement experiment is preregistered in
`denge-game-paired-continuation-preregistration-20260902.md`; this result remains
failed and is used only for variance planning.
