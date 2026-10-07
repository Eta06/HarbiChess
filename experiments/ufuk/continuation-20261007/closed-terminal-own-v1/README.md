# Closed-terminal own-WDL fallback (unexecuted)

This is a separate, prospective data-coverage variant for use only if the
current own-Q/TD routes do not qualify. It has not collected real games, built
real targets, trained a candidate, or run a strength match.

## Frozen collection rule

- Same teacher-selected TRAIN pool, same frozen teacher candidate, and same
  8,192-node / q2 / max-depth-8 search.
- Outcome-blind deterministic order of at most 128 roots. Play complete
  episodes up to 400 plies, stopping only after 1,024 rule-verified terminal
  pre-action rows are collected or the 4,096 actor-row ceiling is reached.
- Training rows are the first 1,024 chronological rows from completed,
  unprotected games. Every row gets exact W/D/L from the terminal board in the
  mover's perspective. Game caps, actor-budget tails, and protected episodes
  are preserved as unknown/excluded records; they are never draws.
- An episode is excluded in full if any played board, static evaluator input,
  or final board intersects the protected placement-alias set.
- Search values are retained only as diagnostics. No Stockfish labels or
  teacher values enter the targets.

## Separate learning contract

The target method is ordinary Monte Carlo terminal WDL regression against the
frozen authoritative-prior logit plus a learned residual. It initializes from
the named same-seed teacher weights only; it creates fresh Adam and RNG state.
The closed-terminal dataset, provenance, native payload, proof, and arena
adapter all have distinct schemas from the existing Q-only and TD(lambda)
routes. The planned budget is 64 fixed updates, with an 8/4/resume-8 synthetic
native proof and six fresh-process payload loads before any fit.

This is standard terminal-return learning, closest to AlphaZero-style
self-play value targets. The hypothesis is only that complete-game coverage
may supply a denser, less bootstrapped value target than the existing
short-horizon Q trace. It is not a novelty claim. The eligible set is
completion-conditioned and therefore favors shorter games; it is not an
unbiased estimate over all actor transitions. Collection may also fail the
registered row target under the hard actor cap. Such a failure must be kept
and reported without increasing the budget or selecting games by outcome.

## Scratch checks

Synthetic tests cover terminal mover perspective, unknown caps, protected
trajectory exclusion, target-contract rejection, and exact native split/resume.
They do not qualify real collection, conversion, CPU runtime, or strength.
