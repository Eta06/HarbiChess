# KOPRU qualified replay transfer preregistration

The first dual-search run is closed as failed and will not train a learner. Its 10,176-position
replay passed every diversity and telemetry-completeness requirement, but failed the old requirement
that more than 55% of individual internal search-Q deltas be positive. A fresh 48-position audit
showed that the stored and reconstructed clean policies were identical (100% top-action agreement,
TV and KL both zero) and that their verified improvement over raw was +0.09759 with a positive 95%
interval of +0.03177 to +0.18089. Per-position internal Q signs are therefore retained as telemetry,
not used as a replay-coverage or teacher-strength gate. Teacher strength remains an independent,
verified bootstrap-confidence gate.

The replacement run is frozen as follows:

- Run ID: `kopru-qualified-replay-20260828-01`
- 96 entirely fresh games, seed `2026082809`, 24 actors, 8 oracle processes
- No replay records from the failed dual-search run are reused
- Behavior root: 64 simulations with 25% root Dirichlet noise
- Target root: independent 64-simulation noise-free depth-1 oracle-bootstrap search
- Maximum 256 plies, schema 12, max-ply outcomes remain unknown rather than draw
- Validation fraction 25%; all continuation/repetition/value-policy/root-halving transforms off
- The replay must pass the unchanged phase, tactical/quiet, WDL/outcome, material/structure,
  uniqueness, sample-volume, and telemetry-completeness coverage gates

If coverage passes, a fresh alignment audit uses 48 stratified validation positions, seed
`2026082810`, 64 simulations, oracle depth 1, verifier depth 4, and 2,000 bootstrap samples. It must
reach at least 95% stored-clean top-action agreement, at most 0.05 mean TV, and strictly positive
lower 95% verified-value bounds for both stored and clean targets versus raw.

Only if both stages pass, one learner transfer uses seed `2026082811`, 200 steps, batch size 64,
learning rate 0.0002, validation every 10 steps, patience 12, and metric-independent snapshot
selection. Every arena-eligible checkpoint must improve legal teacher-policy cross-entropy by at
least 2%, keep known-outcome WDL cross-entropy within 1.02x, keep expected-score ECE regression at
or below 0.02, preserve raw tactical solves, preserve 64/512-search tactical solves, and keep applied
gradient norm at or below 5.0. Failure at any stage blocks arena and promotion. Sample size, gates,
learning rate, exposure, and network capacity will not be changed after observing results.
