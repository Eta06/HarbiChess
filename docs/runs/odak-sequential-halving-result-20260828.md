# ODAK deterministic sequential-halving result

Date: 2026-08-28  
Source commit: `36826cc`  
Artifact: `artifacts/diagnostics/odak-sequential-halving-20260828-01/qualification.json`  
Decision: allocator failed strength/coverage gates; learner, arena, generation, and promotion remain blocked

## Frozen result

ODAK allocated exactly 512 or 800 root search slots through deterministic sequential elimination over at most the top 16 prior actions. It used the unchanged release model, qualified depth-1 oracle teacher, depth-4 verifier, MIHENK 96/48 positions, and frozen worker/batching settings. The run took 189.44 seconds and made 183,364 non-terminal neural backend evaluations.

| Validation metric | ODAK | Gate |
|---|---:|---:|
| 512/800 selected-action agreement | **83.33%** | at least 75%, passed |
| Mean verified delta vs raw | +0.05720 | no more than 0.01 below TERAZI, failed |
| Verified delta 95% interval | [+0.01950, +0.10724] | positive, passed |
| Harmful action ratio | 2.08% | at most 10%, passed |
| Mean verified regret | 0.09787 | at most 0.10, passed |
| Best-action coverage in top-16 prior set | **75.00%** | at least 80%, failed |
| Exact slot accounting | 100% | passed |

TERAZI standard 800 top-Q achieved +0.08822 mean verified improvement. ODAK's +0.05720 is 0.03102 lower, exceeding the frozen maximum regression of 0.01. Sequential halving solved the leading-action stability problem but discarded too much strength.

## Root cause

The broad standard PUCT search visits approximately 99% of legal actions and discovers useful moves outside the policy prior's top 16. ODAK concentrates the same compute and separates its candidate actions more consistently, but the release policy representation fails to place the independently best move inside that candidate set on one quarter of validation positions.

This explains why neither extreme is a qualified teacher:

- broad PUCT preserves action coverage and aggregate strength but produces diffuse, budget-sensitive visit targets;
- top-16 sequential halving produces stable decisions but inherits policy-prior recall failures and loses verified strength.

Changing top-k on this evidence set would be post-hoc tuning and is prohibited. The earlier KILIC result also shows that sparse winner sharpening alone does not transfer into strength.

## Literature alignment

DeepMind's Gumbel planning work does more than sequential halving: it combines root allocation with completed-Q transformations and outputs improved action weights. The official Mctx implementation explicitly recommends Gumbel MuZero when Q values are correctly evaluated and distinguishes its completed-Q transform from ordinary visit normalization. HarbiChess currently has neither completed-Q for unvisited actions nor an action-value/ranking training representation, so its current root-halving heuristic cannot inherit that policy-improvement property.

Relevant primary sources:

- [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO)
- [DeepMind Mctx Gumbel policy implementation](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py)
- [Mctx policy-improvement demonstration](https://github.com/google-deepmind/mctx/blob/main/examples/policy_improvement_demo.py)

## Decision point

The evidence rejects further visit-mixture, threshold, exposure, capacity, and top-k tuning on the current data. The next substantial step should add an explicit action-value/ranking representation trained from stable searched actions, then use that representation to complete values for broad legal support before sequential allocation. This route preserves coverage while giving the allocator a learnable criterion stronger than the current policy prior.

An alternative is to implement full completed-Q Gumbel AlphaZero directly in search. That is shorter but riskier with the present network because completed values for unvisited actions would still be derived from a policy/value model whose action recall is the demonstrated blocker. The recommended order is therefore action-value representation first, controlled offline transfer and calibration qualification second, then completed-Q root allocation. This requires a new schema/head/loss contract and should be treated as a deliberate architecture stage rather than another target heuristic.
