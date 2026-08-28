# DEGER action-value representation result

Date: 2026-08-28  
Source commit: `f83418b`  
Artifact: `artifacts/runs/deger-action-value-20260828-01/result.json`  
Decision: transfer failed; completed-Q search, learner continuation, arena, generation, and promotion remain blocked

## Frozen result

DEGER added a zero-initialized dueling action-value head and trained only that head for exactly 480 steps on the frozen TERAZI 96 train rows. The release trunk, policy head, and WDL head were detached and excluded from optimization. Validation used the separate 48 TERAZI rows and fresh depth-4 action verification.

| Step | Validation Q MSE | Teacher-Q Spearman | Verified delta | Harmful | Regret | Top-16 best coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.09565 | 0.000 | -0.00254 | 6.25% | 0.15762 | 68.75% |
| 60 | 0.11522 | -0.035 | +0.00125 | 18.75% | 0.15382 | 81.25% |
| 120 | 0.11208 | -0.048 | -0.00274 | 25.00% | 0.15782 | 79.17% |
| 240 | 0.11488 | -0.071 | -0.00761 | 25.00% | 0.16269 | 81.25% |
| 480 | 0.11440 | -0.048 | -0.01912 | 22.92% | 0.17419 | 79.17% |

No non-zero checkpoint passed. Losses and gradients were finite, with maximum norm 0.3191. The maximum policy/WDL logit change was exactly zero at every checkpoint, and tactical behavior remained unchanged, proving that the isolation mechanism worked.

## Diagnosis

The head used a convolution followed by a global `256 → 4,672` action matrix, adding roughly 1.2 million trainable parameters. Ninety-six labelled positions cannot identify this nearly position-specific mapping. Validation error increased immediately and action ranking became negatively correlated with teacher Q. More steps or width tuning on these rows would amplify memorization rather than repair representation.

HarbiChess actions already have a structured `64 squares × 73 move planes` geometry. The next representation should exploit it directly: a spatial `1×1 convolution → 73 action planes` shares parameters across board squares and reduces the action head to approximately 4–5 thousand parameters. It should be tested on a completely fresh, non-overlapping Q-labelled train/validation set with the unchanged DEGER gates. The global-linear head, its duration, and its exposure are frozen and rejected.
