# KANIT v7 arena pre-registration — 2026-08-26

## Frozen models

No training, replay, continuation exposure, policy target, temperature, or checkpoint
selection may change during this evidence run.

| Arm | Checkpoint | Model SHA-256 |
| --- | --- | --- |
| Continuation-off | `candidate-step-000130` | `10be51c4b169c6964cf7db2eccb22f57b48a9bbc243abe4868209ffc100d8712` |
| V7 continuous regret | `candidate-step-000110` | `041ead5c01165ee0e0148fa35038d7bd19521aff79f9b2738355cb627970b6a3` |

The hashes were independently verified against the checkpoint files before any new
arena game was played.

## Fixed additional evidence

- 400 additional games per arm from 200 color-balanced opening pairs.
- Seed: `2026082612`.
- Opening plies: 12.
- MCTS simulations: 32.
- Maximum game length: 256 plies.
- Workers: 96.
- Inference wait: 0.25 ms.
- Existing independent ARALIK evidence: 200 games per arm.
- Final combined evidence: exactly 600 games per arm.

The additional sample count will not be extended or shortened after observing interim
results. The new 400-game result must be combined with all 200 existing ARALIK games;
neither block may be selectively excluded.

## Sample-size rationale

The first 200 fresh paired observations had score-difference standard deviation
`0.30038`. A combined sample of 600 has approximately 80% power for a practically
meaningful paired strength difference near +3.4 percentage points and higher power for
the +4.5-point effect observed during KASIM. Establishing a +1.5-point effect with high
power would require several thousand games and is outside this evidence budget.

## Frozen combined gate

The same deterministic 50,000-resample paired bootstrap and behavioral thresholds are
used. Promotion requires every condition:

1. Paired score difference: two-sided 95% lower bound greater than zero.
2. Avoidable-threefold difference: candidate point estimate no worse than control and
   one-sided 95% upper bound at most +5 percentage points.
3. Win-rate difference: candidate point estimate greater than control and one-sided
   95% lower bound no worse than -2 percentage points.
4. Decisive conditional score: candidate point estimate must not regress.

Bootstrap seed remains `2026082603`. Thresholds and sample size cannot change after
the new arena starts. Passing permits promotion/new-generation evaluation; it does not
silently promote before checkpoint, artifact, dashboard, and champion-chain integrity
checks complete. Failure leaves the champion and generation unchanged.
