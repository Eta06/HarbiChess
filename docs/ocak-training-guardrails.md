# OCAK training guardrails

OCAK establishes the data and learner invariants required before recursive
self-learning begins. It deliberately does not promote checkpoints or start a
large training run.

## Value perspective

Every value target is expressed from the side to move in the recorded state.
`root_value` is the search estimate from that perspective and `outcome_value`
is the final game result transformed to the same perspective. The WDL learner
maps `+1`, `0`, and `-1` to win, draw, and loss classes. MCTS negates a leaf
value once per parent edge during backpropagation. Tests cover terminal-result
antisymmetry, alternating backpropagation signs, and a mate-in-one transition.

## Replay compatibility and integrity

Replay shards carry independent replay, board-encoder, action-space, and target
schema versions. A reader rejects a shard unless all versions exactly match the
runtime. Each gzip JSONL shard also records its source commit, checkpoint,
generation, split, game/sample counts, and a SHA-256 digest of the canonical
record payload. Shards are written to temporary files, flushed, and atomically
renamed only after completion.

Train/validation assignment hashes the stable game ID. The whole game goes to
one split, so positions from a single trajectory cannot leak across the
boundary. Replay sampling chooses games uniformly before choosing a position;
long games therefore do not dominate merely because they contain more plies.

## Collapse monitoring

Self-play batches report duplicate-game ratio, unique normalized-position
ratio, selected action-space coverage, mean visit-policy entropy and effective
branch count, mean game length, W/D/L counts, and opening-prefix entropy at
plies 4, 8, and 12. No single metric proves diversity. Regressions across
opening entropy, effective policy branches, and duplicate trajectories are the
primary collapse signal; action-space coverage is a slower supporting signal.

## Learner pilot gate

The MLX learner uses soft visit-policy cross-entropy plus side-to-move WDL
cross-entropy. Losses and gradients must be finite before the optimizer can
update, and gradients are norm-clipped. A small pilot requires disjoint,
non-empty train and validation games, a configured minimum training-loss
improvement, and a maximum validation-loss degradation ratio. A failed gate
blocks a larger run instead of silently continuing.

## Stop and resume

An immutable checkpoint directory contains model weights, optimizer state, and
the game-balanced sampler RNG state. Its resume manifest also records counters,
replay cursor, elapsed training time, source commit, and SHA-256 checksums for
every artifact. Publication uses one directory rename. Loading verifies every
artifact before mutating the learner and restores the exact next sampling and
optimizer trajectory.

## DEVIR boundary

OCAK produces candidates but never changes the champion. DEVIR will run a
color-balanced candidate-versus-champion arena and promote only when the
confidence rule passes. A failed candidate leaves the champion pointer
untouched. The next candidate should restart from the champion while retaining
compatible champion-generated replay; failed-candidate weights should not form
an implicit champion chain.
