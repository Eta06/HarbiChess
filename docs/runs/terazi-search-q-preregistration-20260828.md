# TERAZI search-Q reliability preregistration

Date: 2026-08-28  
Parent evidence: DENGE and UYUM soft visit targets both failed 20% row qualification  
Decision scope: decide between Q-derived target work and search-teacher repair; no learner, arena, generation, or promotion

## Hypothesis

The depth-1 bootstrap search selects stronger actions in aggregate, but normalized visits remain too diffuse to form a sufficiently broad verified learner target. Root child-Q may contain a cleaner action ranking than visits because visit counts include PUCT exploration. Before designing a Q-derived target, TERAZI must prove that child-Q is independently correct and stable across compute budgets.

## Frozen execution

Use the unchanged release checkpoint and exact MIHENK 96 train plus 48 validation positions. Run clean 512- and 800-simulation PUCT with the qualified depth-1 process oracle, 24 root workers, eight oracle workers, batch cap 48, batching wait 0.25 ms, and seed `2026082820`. Evaluate every legal root action with the unchanged deterministic depth-4 verifier.

For each budget and position report, over actions with at least one visit:

- Spearman rank correlation between child-Q and independent verified value;
- top-Q and top-visit action, verified delta from the raw-policy argmax, and verified regret from the best legal action;
- visit-weighted Q calibration MAE against verified values;
- number and fraction of legal actions visited.

Across 512 and 800 report top-Q agreement, Q-ranking correlation on their common visited support, and absolute Q drift. Unvisited actions cannot be assigned Q=0 and cannot participate in Q ranking or Q-derived labels.

## Frozen decision gate

Q-target development is authorized only if validation satisfies all conditions:

- mean 800-budget Q/verified Spearman correlation at least `0.35`;
- bootstrap 95% lower bound of the mean 800-budget top-Q verified delta versus raw is strictly positive;
- 800-budget top-Q harmful-action ratio at most `10%`, where harm is delta at most `-0.025`;
- mean 800-budget top-Q verified regret at most `0.10`;
- 512/800 top-Q agreement at least `75%`;
- mean cross-budget Q Spearman correlation at least `0.70`;
- the top-Q mean verified delta is no worse than top-visit by more than `0.01`.

Bootstrap uses 2,000 resamples. Thresholds, positions, and budgets cannot change after results become visible.

If the gate passes, the next preregistered experiment may construct a soft Q-improvement target while preserving multiple near-tied actions. If it fails, another visit/Q target transform is prohibited; work returns to search allocation, leaf evaluation, or action-value supervision. Learner and generation remain blocked in either case until a later target itself qualifies.
