# KRITIK frozen joint policy/value transfer result

Date: 2026-08-30

Artifact: `artifacts/runs/kritik-joint-policy-value-20260830-01/result.json`

## Decision

KRITIK failed its preregistered Stage-A value-head learnability gate. Joint training, continuation
ranking, Full Gumbel tactical comparison, arena, continuous learning, generation, and promotion did
not run.

## Provenance

- baseline SHA-256 matched `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`;
- both shards used corrected target schema 12;
- train and validation had disjoint games;
- 7,424 train and 1,792 validation max-ply/unknown rows were excluded from value supervision;
- 7,698 train and 2,340 validation terminal-labelled rows remained;
- decisive labels alternated correctly with side to move; draw labels were zero throughout each
  game.

## Frozen value warmup

The value-only control used outcome-balanced, then game-balanced sampling. The value head was the
only trainable component. The best validation checkpoint was step 20; validation early stopping
ended the run at step 140.

| Metric | Baseline | Selected step 20 |
| --- | ---: | ---: |
| validation micro WDL CE | 1.09893 | 1.10194 |
| validation macro WDL CE | 1.09888 | 1.10220 |
| validation Brier | 0.66687 | 0.66890 |
| validation Pearson | 0.08515 | -0.39830 |
| train Pearson | 0.08515 baseline validation reference | 0.25868 |
| validation loss-to-draw margin | 0.00044 | -0.00274 |
| validation draw-to-win margin | -0.00015 | -0.00443 |

Maximum gradient norm was finite at `0.73768`. Later checkpoints systematically worsened held-out
CE and Brier, so longer training would amplify overfit rather than repair the signal.

## Interpretation

Removing unknown max-ply rows from the sampler and balancing terminal outcomes did not make the
current KOPRU value target generalize across games. The opposite-sign train and validation Pearson
shows that the head can fit a small trajectory collection but does not learn a transferable state
value. The effective number of independent labelled units is the number of terminal games, not the
7,698 highly correlated per-ply rows.

The next step is the preregistered structural audit: separate insufficient independent games and
high-variance early-ply Monte-Carlo labels from head/representation failure using matched-exposure
position-leakage, late-ply, game-disjoint, and shuffled-label controls. No threshold is relaxed.
