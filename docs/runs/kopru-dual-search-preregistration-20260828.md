# KOPRU dual-search replay preregistration

The clean-target/single-search actor run is closed without learner training. It produced 7,086
positions from 48 games, 37 threefold repetitions, and failed replay volume plus winning-state
coverage. The target was clean, but deterministic behavior after ply 30 collapsed trajectories.

The dual-search sanity keeps the previously diverse behavior policy and independently stores the
qualified clean teacher target:

- Run ID: `kopru-dual-search-sanity-20260828-01`
- 48 fresh games, seed `2026082807`, 24 actors, 8 oracle processes
- Behavior root: 64 simulations with the existing 25% root Dirichlet noise
- Target root: separate 64 simulations, noise off, same network and depth-1 value teacher
- The behavior move may be outside the clean target support and is explicitly versioned in schema 12
- Maximum 256 plies, first 30 plies sampled by temperature 1, then noisy behavior argmax
- Validation fraction 25%; all continuation/repetition/value-policy/root-halving transforms off
- Learner disabled until replay coverage and target alignment pass

The default replay coverage thresholds stay fixed. The alignment audit uses 48 stratified validation
positions, seed `2026082808`, and the existing target-integrity gates: at least 95% stored-clean top
agreement, TV at most 0.05, and positive lower 95% verified-value bounds for both stored and clean
targets versus raw.

If both gates pass, exactly one legal-masked learner transfer uses the frozen 200-step configuration
and existing policy/WDL/calibration/raw-tactical/search-tactical gates. Failure blocks arena. No
sample-size, threshold, LR, exposure, or network-capacity change is allowed after results.
