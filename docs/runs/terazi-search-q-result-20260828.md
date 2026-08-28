# TERAZI search-Q reliability result

Date: 2026-08-28  
Source commit: `1494d7b`  
Artifact: `artifacts/diagnostics/terazi-search-q-20260828-01/q-reliability.json`  
Decision: Q-target gate failed; learner, arena, generation, and promotion remain blocked

## Frozen result

The run repeated clean 512/800 search on the exact MIHENK 96 train and 48 validation positions and evaluated every legal root action with the independent depth-4 verifier. It took 203.99 seconds and evaluated 174,812 neural positions. No unvisited action was assigned an artificial Q value.

| Validation metric | Result | Gate |
|---|---:|---:|
| 800-Q vs verified Spearman | 0.4420 | at least 0.35, passed |
| Top-Q verified delta | +0.08822 | positive interval, passed |
| Top-Q delta 95% interval | [+0.03401, +0.14797] | passed |
| Top-Q harmful ratio | 8.33% | at most 10%, passed |
| Top-Q verified regret | 0.06686 | at most 0.10, passed |
| Cross-budget Q Spearman | 0.80685 | at least 0.70, passed |
| Top-Q delta minus top-visit | +0.00503 | no regression, passed |
| 512/800 top-Q agreement | **64.58%** | at least 75%, failed |

Q is better correlated with verified action value than visit count (0.4420 versus 0.4040), and its selected action is slightly stronger with lower regret. However, its leading action still changes too often between 512 and 800 simulations. The preregistered Q-target gate therefore fails despite six other conditions passing.

## Allocation diagnosis

At 800 simulations, search visited a mean 99.01% of legal root actions. The mean cross-budget absolute Q drift was only 0.0130, so instability is dominated by near-tied leading actions rather than wholesale Q-ranking failure. Standard PUCT spends enough budget on broad exploration that the leading alternatives are not reliably separated.

TERAZI prohibits a post-hoc Q target after this failure. The next work returns to search allocation: test whether the existing sequential-halving/forced-continuation mechanism concentrates the same fixed root compute on leading actions and raises top-Q or set-level stability without losing independent verified strength. Learner experiments remain closed.
