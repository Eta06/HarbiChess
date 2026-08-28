# ODAK deterministic sequential-halving qualification preregistration

Date: 2026-08-28  
Parent evidence: TERAZI Q reliability passed six of seven gates but 512/800 top-Q agreement was 64.58%  
Decision scope: fixed-budget search allocation diagnostic; no learner, arena, generation, or promotion

## Rationale

TERAZI found that standard 800-simulation PUCT visits 99.01% of legal root actions. Its Q ranking is independently useful, but the leading near-tied alternatives are not separated reliably. Fixed-budget best-arm identification should spend root compute on reducing simple regret rather than continuing broad cumulative-regret exploration.

ODAK follows deterministic sequential halving at the root, with ordinary PUCT inside each forced continuation. This is materially different from the earlier KILIC heuristic: KILIC spent only 3/7 simulations on four preselected visit leaders and changed 0.38% of targets. ODAK considers up to 16 prior actions and allocates the complete 512/800 root budget through repeated elimination rounds.

The design follows the root-allocation principle in Gumbel AlphaZero/MuZero and DeepMind Mctx, but uses Gumbel scale zero because this is clean deterministic qualification. It does not claim the policy-improvement guarantee of full Gumbel search; HarbiChess does not yet implement Mctx's completed-Q transformation.

## Frozen algorithm

For each position and total search-evaluation-slot budget `B` in `512, 800`:

1. Evaluate the root once and take at most the 16 highest-prior legal actions.
2. Let `R = ceil(log2(K))` for `K` considered actions.
3. In each round, allocate `floor(remaining_evaluations / (active_actions * remaining_rounds))` evaluations to every active action. One evaluation initializes the forced child root; the rest are clean continuation MCTS simulations.
4. Combine an action's independent round estimates by evaluation-count-weighted mean.
5. Rank by mean continuation value, then prior, then UCI; retain `ceil(active/2)` actions.
6. Spend any final remainder on the surviving action and return the highest accumulated-value action.

All evaluation slots, including root and forced child expansions, count against `B`; no run may exceed it. A terminal node consumes its allocated slot even though it correctly requires no neural backend call. This convention prevents terminal-heavy positions from receiving extra non-terminal search compute while keeping the allocator exactly budget matched. Search uses the unchanged depth-1 oracle, no root noise, 24 position workers, eight oracle workers, batch cap 48, wait 0.25 ms, seed `2026082821`, and the exact MIHENK 96/48 positions. Every legal root action retains the unchanged depth-4 verifier value from a fresh deterministic evaluation.

## Frozen gate

ODAK is qualified for a later soft-target experiment only if validation satisfies:

- 512/800 selected-action agreement at least `75%`;
- 800-budget selected-action verified delta versus raw has a strictly positive bootstrap 95% lower bound;
- harmful selected-action ratio at most `10%` for deltas at or below `-0.025`;
- mean selected-action verified regret at most `0.10`;
- mean selected-action verified delta is no worse than TERAZI standard 800 top-Q by more than `0.01`;
- the considered top-16 prior set contains the independently best legal action in at least `80%` of rows;
- allocated search-evaluation slots equal the requested budget for every row; measured backend calls are reported separately and may be lower only for terminal nodes.

Bootstrap uses 2,000 resamples. Thresholds and sample sizes cannot change after results become visible. A pass authorizes only a separately preregistered uncertainty-preserving target qualification; learner remains blocked. A failure rejects this allocator and returns the investigation to value/action representation rather than tuning K, budgets, or gates on this position set.
