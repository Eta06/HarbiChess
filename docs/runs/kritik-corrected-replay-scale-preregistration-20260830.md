# KRITIK corrected-replay scale preregistration

## Decision being tested

The frozen value head partially memorizes positions when the same games occur in train and
validation, but it does not generalize to held-out games. This experiment tests whether the
blocker is the number of independent terminal trajectories rather than training duration or
search allocation.

No new self-play, search allocation change, learner generation, arena, promotion, or champion
mutation is authorized by this experiment.

## Frozen inputs

- Baseline SHA-256: `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`.
- Replay: every on-disk schema 10+ shard produced from that exact baseline in the eight
  `KOPRU`/`OMURGA` runs enumerated by the implementation.
- Unknown/max-ply rows remain excluded from WDL supervision.
- Games are identified by their complete root and move trajectory, not by run-local game ID.
- Duplicate trajectories are retained once only and can never cross the train/validation split.
- Split: deterministic, outcome-stratified 75/25 split of trajectory fingerprints.
- Trunk and policy head remain bitwise frozen; only the existing WDL head is trainable.
- Outcome-balanced and game-balanced sampling remains in force.
- Optimizer: Adam, learning rate `5e-4`, batch size `64`, no weight decay.
- Exposure: `400` steps, validation every `20` steps, fixed seed `2026083049`.
  This is fixed before results and provides comparable per-independent-game exposure without
  interpreting a longer curve after the fact.

## Frozen gates

All gates must pass together on held-out trajectory fingerprints:

- macro WDL cross-entropy improvement at least `0.10` over the frozen baseline;
- Brier improvement at least `0.03`;
- expected-score Pearson correlation at least `0.20`;
- ordered outcome means with loss-to-draw and draw-to-win margins each at least `0.03`;
- the frozen non-value parameter hash must remain exact.

The selected checkpoint is the lowest held-out macro WDL cross-entropy checkpoint. A partial
improvement is evidence for diagnosis only and is not a pass.

## Interpretation fixed before results

- Pass: independent-game scale was a material blocker. Proceed to the preregistered joint
  policy+value transfer, still requiring imitation, tactical, calibration, continuation-ranking,
  and search-strength gates.
- Clear improvement but failed frozen gates: data scale is directionally useful but insufficient;
  audit representation/head and target variance before producing new games.
- No held-out improvement: stop scaling this target unchanged and move directly to a controlled
  representation/head/loss audit.

