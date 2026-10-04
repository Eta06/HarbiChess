# UFUK: Q-range floor mechanism gate failed

Neither0.05 nor0.20floor passed the preregistered selected-move and soft-target
quality gates. Historical1e-8 FullGumbel default retained; no arena/training or
model promotion. Main Stockfish/AlphaZero-level strong/fast/self-learning goal
remains unmet. This conventional ablation is not claimed as novel.
[Protocol](UFUK-q-range-preregistration-20261004.md),
[primary Mctx audit](../research/UFUK-search-scaling-audit-20261004.md).

Clean source54ae3e5ba56c36910ca08829602bcd5ddcc04fbf; original18f model and frozen128full-history
native development positions/16families/3433all-legal independent references.
No new engine calls, labels or gradients. All three arms16sim/max4/Gumbel0,
value_scale0.1/maxvisit_init50/CPU1FP32/eagerB1, seed20261018/rotatingarm order.
6477evaluation calls,14.199346s whole diagnostic, below7500calls/300s caps.
Repeated default control matched all128 historical max4selected moves, visit
counts and complete policies exactly; those repeats are not new independent
samples. Full reference hashes/histories/legal support/16visits/expectations
verified. Python counter counts evaluator calls, not a GPU utilization metric.

| Floor | Endpoint vsdefault | Mean16family effect | Four-contrast adjusted98.75% bootstrap | Gate |
| --- | --- | --- | --- | --- |
|0.05|selected_expected_score|-0.006312500|[-0.024312500000000004, 0.0]|failed|
|0.05|soft_expected_score|-0.001812570|[-0.0059079152343869375, 0.0009932203258315989]|failed|
|0.2|selected_expected_score|-0.003828125|[-0.023625000000000004, 0.00853125]|failed|
|0.2|soft_expected_score|-0.004172534|[-0.011970081899928191, 0.00492841036601686]|failed|

Both endpoints need>=0.01mean and adjustedlower>0; no floor qualified.
10000whole-family resamples/seed18; Bonferroni four contrasts (two floors×two
endpoints), conditional nominal95% and conservative Hoeffding bounds also
recorded. Reused development positions and prior method selection limit
inference; no independent confirmatory significance/Elo. Negative averages do
not establish a universal disadvantage or identify the cause of learning failure.

Original search/signs/rules/policy/value semantics unchanged, subclass is
experiment-only. Unit tests verify exact default outputs, real terminal-mate
backup and budget, small-range damping and unchanged large ranges. Fullsuite
512passed/zero skips43.79s; no AppleMetal/CUDA device claim. Forty-ply24fresh
arena roots frozen before diagnosis but unused after qualification failed.
All failed diagnostics/source/tests retained in
[exact evidence](UFUK-q-range-evidence-20261004.json), models remain separate.
No paid compute and no completed new remote binary-backup claim.
