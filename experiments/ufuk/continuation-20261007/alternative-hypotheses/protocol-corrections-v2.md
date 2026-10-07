# Corrections and one secondary policy-head option

This addendum is separate from `own-search-next-mechanisms.md`; the original file is preserved.

## Corrections to the evidence summary

- The independent own-Q audit covered **1,024 selected histories per seed** (2,048 total). Its depth histograms are per seed; they are not 2,048 histories per seed. The immutable receipt is `experiments/ufuk/cpu-own-q-residual18-v1/evidence/ownQ2048-ROOT-history-and-six-search-result.json`.
- Two distinct 18-feature stages must not be conflated. The older CLASSIC stage completed 160 games and failed: learned-versus-SF512 scores `.0625/.03125`, prior `.125/.125` (`experiments/ufuk/cpu-classical-own-v1/arena/independent-fullhistory-result.json`). The later own-Q residual C18 stage also completed 160 games and failed: `.09375/.03125`, prior `.125/.125` (`experiments/ufuk/wrap-20261006/C18-fullhistory-independent-result.json`). The later seed05 count was one win, one draw and fourteen losses: 1.5/16=.09375. ROOT reread both immutable full-history receipts on Oct7. Neither stage was profile-only or virgin confirmation.
- The fresh E8-derived MC, SC, and full-critic screen failures are already summarized in the original note. All screens are limited development evidence and none is an estimate of final strength.

## TD(lambda) prototype boundary

`tdlambda-prototype-v1/targets.py` is a dormant, pure target transformer. Its inputs are final per-episode row lists from the future frozen own-Q labels packet, not raw event logs. It does not load or transform the current sealed collection; it performs no search, model call, optimizer step, or strength evaluation. Lambda is fixed at `.5`. It records source row IDs and a target receipt without changing source rows.

The transformer uses `clipped_q_mover`, converts to a common White-outcome frame, recurses along consecutive actually played rows, and converts each output back to that row's mover perspective. A completed own-terminal game replaces the last row target with its exact W/D/L result. An UNKNOWN episode ends with the last recorded **pre-action** own-Q as the bootstrap; the code never claims a value for the unrecorded board after the last action. Entire protected episodes are excluded. It rejects gaps, mixed status, invalid movers, or incomplete known WDL labels.

The pure tests cover one- and two-ply episodes, both mover colors, exact draws, open UNKNOWN tails, and protected-episode exclusion. The prototype is not an approved training pipeline; root must bind it to a sealed producer receipt and freeze row masks, sample counts, and update counts before any fit.

## Optional action-head direction

An action head trained on the current collector's `selected_best_uci` could help move ordering, but it would merely imitate a single search choice from the same frozen parent. The current stream does not record visit distributions or per-action root values, so this would be weaker than AlphaZero-style policy improvement targets. It needs a distinct schema and an ablation where both parent and candidate use the same policy-enabled search implementation; comparing a policy-guided candidate to a differently configured value-only parent would confound learned weights with changed search. Giraffe already combines learned evaluation/policy representations with chess search; Lc0 and AlphaZero use policy priors to guide tree search; Expert Iteration/Gumbel AlphaZero cover search-to-policy improvement. I would defer this until the value/TD and equal-node target experiments isolate the current bottleneck.

This is not novel. Baxter et al.'s TDLeaf(lambda), AlphaZero/MuZero n-step targets, Giraffe, and Gumbel/Expert Iteration are established related methods. The only proposed work is a small, outcome-safe implementation and controlled test in this codebase.

Primary references: Lai, [Giraffe: Using Deep Reinforcement Learning to Play Chess](https://arxiv.org/abs/1509.01549); [Leela Chess Zero's AlphaZero/search overview](https://lczero.org/dev/lc0/search/alphazero/).
