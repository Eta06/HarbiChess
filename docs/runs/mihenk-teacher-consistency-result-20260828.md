# MIHENK teacher consistency result

Date: 2026-08-28  
Source commit: `8fa3f6b37ad10f63afe03488ec892e8c4666334c`  
Artifact: `artifacts/diagnostics/mihenk-teacher-consistency-20260828-01/consistency.json`  
Decision: consensus gate failed; learner ablation, arena, and generation remain blocked

## Frozen audit execution

The audit used the preregistered release checkpoint, 96 stratified train records, 48 separately stratified validation records, clean search budgets `64, 128, 256, 512, 800`, depth-1 process teacher oracle, and independent depth-4 process verifier. No thresholds, sample counts, or search settings changed after results became visible.

Wall time was 254.65 seconds. MLX evaluated 237,079 positions in 49,251 batches (mean batch 4.81, largest 16), with 185.50 seconds of backend time. During the high-budget search, eight teacher-oracle processes were around 80% CPU each and AGX device utilization was approximately 63%.

## Classification

| Partition | Stable/high-confidence | Ambiguous/budget-sensitive | Harmful |
|---|---:|---:|---:|
| Train, 96 | 8 (8.33%) | 84 (87.50%) | 4 (4.17%) |
| Validation, 48 | 2 (4.17%) | 45 (93.75%) | 1 (2.08%) |

The validation stable ratio failed the preregistered 20% minimum. The harmful ratio passed the 10% maximum, showing that the dominant problem is not systematically bad actions; it is instability and insufficient separation.

The two stable validation rows had a large verified improvement mean (`+0.5243`, bootstrap interval `[+0.2509, +0.7977]`), but two rows are not enough to authorize learner conclusions.

## Cross-budget consistency

Validation policy comparisons:

| Budgets | Top-action agreement | Mean TV | Mean JSD |
|---|---:|---:|---:|
| 64 vs 128 | 68.75% | 0.2165 | 0.0703 |
| 64 vs 256 | 60.42% | 0.3303 | 0.1298 |
| 64 vs 512 | 35.42% | 0.3978 | 0.1746 |
| 64 vs 800 | **25.00%** | **0.4284** | **0.1969** |
| 128 vs 256 | 79.17% | 0.1758 | 0.0464 |
| 256 vs 512 | 64.58% | 0.1487 | 0.0288 |
| 256 vs 800 | 37.50% | 0.1992 | 0.0460 |
| 512 vs 800 | **66.67%** | 0.0926 | 0.0100 |

The 512-versus-800 agreement failed the preregistered 75% gate. Even though their distributions are relatively close in TV/JSD, many leading actions are near-tied and swap order as budget increases.

The most common validation ambiguity reasons were:

- top action changed across budgets: 38 rows;
- verified improvement below the stable minimum: 33 rows;
- normalized high-budget visit margin too small: 32 rows;
- high-budget policy TV above the limit: 16 rows.

## Verified strength

The 800-simulation action was independently better than the raw-network action in aggregate:

- validation mean verified delta: `+0.08319`;
- validation bootstrap 95% interval: `[+0.03642, +0.14328]`;
- train mean verified delta: `+0.09308`;
- train bootstrap 95% interval: `[+0.06149, +0.12988]`.

This reconciles the earlier observations. The teacher is stronger on average, but it does not present one stable per-position argmax target at the current budgets. A learner minimizing soft policy cross-entropy can therefore move toward the distributions while losing top-action agreement and raw tactical behavior.

## Gate decision

The gate failed for exactly two preregistered reasons:

1. validation stable-target ratio was 4.17%, below 20%;
2. 512-versus-800 top-action agreement was 66.67%, below 75%.

The three-arm learner ablation was not run. Repeating two stable validation examples would create an easy memorization result, and lowering the thresholds after seeing the distribution would invalidate the experiment.

## Next hypothesis

Do not increase model capacity, learner duration, or stable-target exposure. The next experiment should change the target representation rather than pick a brittle argmax: preserve high-budget policy uncertainty and test whether a consensus distribution (for example, a fixed mixture of independently clean 512/800 visit distributions) is more reproducible across repeated search while retaining positive verified improvement. That proposal needs a fresh preregistration and teacher qualification before any learner run.
