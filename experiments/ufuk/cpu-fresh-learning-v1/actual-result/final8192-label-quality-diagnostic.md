# Read-only own-label diagnostic: fresh E0 8192-action journals

Scope is the two frozen E0 actor journals (one CPU thread). No model forward, optimizer step, new game, or arena query was performed. The stored E8 anchor probabilities, stored Q512 root scores, game terminal results, and known-row identities were read from the accepted journals and the independent readiness receipt. This is descriptive data analysis, not validation on independent positions or evidence of strength.

The readiness receipt SHA is `90162446656105007ca713c17b01f4e0a762d694a179a5be815e0ba6174bae51` and its source-bound journal SHAs are seed 20262805 `94f193481ee71f0d21f55d8addb6e15a46ede4522aaeebf8c2557cc689c9899d` and seed 20262806 `a7b135ea5b835fe7a397fd639147f2a58bad9f2e02c76cae8feb9895a71b1352`.

## Data composition

| Seed | Completed games (1-0 / 0-1 / draw) | Known action rows | Active UNKNOWN actions | Unique complete trajectories | Action count min / median / max |
|---|---:|---:|---:|---:|---:|
| 20262805 | 89 (37 / 45 / 7) | 8,141 | 51 | 85 | 18 / 80 / 291 |
| 20262806 | 82 (32 / 39 / 11) | 8,188 | 4 | 74 | 49 / 85 / 280 |
| Total | 171 (69 / 84 / 18) | 16,329 | 55 | 159 | — |

There were no capped games. The unfinished actor tails remain UNKNOWN and are absent from the outcome metrics. Earlier independent reconstruction found no protected-position exclusions and removed 320 and 640 duplicate rows respectively before train/validation eligibility.

Search completion depth is concentrated at 1–3: depth counts are 1: 2,975; 2: 11,083; 3: 2,189; 4: 58; 5: 4; 6: 2; 8: 18. Thus 99.50% of known rows ended at depth 1–3; 99.89% used all 512 nodes. Exploration selected a random legal action on 799 known rows (4.89%).

## Stored predictors versus eventual mover return

For each known pre-action row, `z` is +1 for a mover win, 0 for draw, −1 for mover loss. The Q predictor is stored root-mover search value clipped to the trainer’s `[-1,1]` target. The E8 predictor is stored `P(win) − P(loss)` from the same mover’s perspective. Rows are correlated within games, so these values are descriptive only.

| Predictor | Row MSE to `z` | Pearson correlation | Correct sign on decisive rows |
|---|---:|---:|---:|
| Clipped own Q512 score | 0.8234 | 0.4638 | 0.7256 |
| Frozen E8 anchor `P(win)-P(loss)` | 0.8300 | 0.4517 | 0.7166 |

The Q target has a small in-sample edge on these aggregate measures; it is not a large or independent signal. Game-equal MSE is 0.8555 for clipped Q and 0.8706 for the anchor. The anchor’s three-class argmax accuracy is 0.5689, cross-entropy 1.8226, and multiclass Brier score 0.6796.

## Validation-split limitation

The registered split hashes complete trajectories, but every trajectory begins at the same ordinary start. Exact identical full-history prefixes therefore cross the train/validation trajectory split: 95 distinct history keys, appearing 3,224 times in train rows and 423 times in validation rows across the two seeds. This does not leak the eight protected known-game positions (the independent audit found zero protected games), and it does not compromise the separate arena test. It does make this internal validation less position-independent than the trajectory-group wording alone may suggest.

## Diagnostic conclusion

I found no terminal-perspective, cap-label, stored-anchor normalization, or history-alignment error in the prior independent journal/data reconstruction. The outcome labels are roughly balanced across mover wins/losses, with fewer draws. The clearest data limitation is narrow, highly repeated opening context plus mostly depth-1–3 search targets; the clipped Q score is only moderately predictive of eventual mover return. If another fit also fails, the next controlled hypothesis should first test broader own-play state coverage while keeping terminal labels, unknown handling, and the registered strength gates fixed. This report does not claim such a change will improve strength or that the current residual/full-critic fits have failed.
