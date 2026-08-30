# KRITIK value-signal structural audit preregistration

Date: 2026-08-30

## Question

The corrected and balanced KRITIK value-head control learned train correlation but reversed sign on
game-disjoint validation. This audit separates four explanations before changing network capacity,
loss weights, or training duration:

1. the apparent 7,698 labelled train rows represent too few independent game outcomes;
2. early-ply one-game Monte-Carlo outcomes have excessive variance for this replay scale;
3. the current value head cannot represent even stronger late-game signal;
4. an unnoticed split/label leakage makes a nominal validation improvement meaningless.

## Frozen descriptive audit

For each split, report terminal/unknown game counts, terminal rows per independent game, outcome and
phase/ply distributions, material-score-to-outcome Pearson, stored root-value-to-outcome Pearson,
and train/validation differences. All statistics retain side-to-move perspective. Unknown max-ply
rows remain excluded.

## Frozen matched-exposure controls

Every arm starts from the exact KOPRU baseline and trains only the existing value head. Trunk and
policy parameters remain frozen. Optimizer, outcome/game-balanced sampling, learning rate `5e-4`,
batch 64, 140 steps, validation every 20 steps, and seed `2026083037` are identical.

- `game-disjoint-all`: original known train and validation rows;
- `position-split-all`: deterministic 75/25 position split of all known rows, deliberately allowing
  the same game in both sides as a leakage/memorization diagnostic;
- `game-disjoint-late32`: only the last 32 labelled positions of every terminal game, retaining the
  original game-disjoint split;
- `game-disjoint-shuffled`: original split with one deterministic permutation of terminal results
  across train games while preserving each game's internally consistent alternating perspective;
  validation labels stay real.

Each arm selects its minimum macro validation WDL CE checkpoint. Report micro/macro CE, Brier,
ECE-10, expected-score Pearson, outcome mean margins, and exact frozen-parameter hashes.

## Interpretation fixed before results

- Only `position-split-all` improves: effective independent-game scale/generalization is the primary
  blocker; more correlated plies or longer training are not a fix.
- `game-disjoint-late32` improves while `game-disjoint-all` does not: early-ply Monte-Carlo target
  variance is the primary blocker; value targets need horizon/bootstrapping or ply-aware weighting.
- Neither real-label arm improves but `position-split-all` memorizes: representation plus small game
  count both remain plausible and require a head/target matrix on fresh games.
- `game-disjoint-shuffled` improves materially: audit for leakage or spurious game identity before
  any learner work.
- No arm improves, including position leakage: existing value head/optimizer representation is the
  immediate blocker.

An arm counts as a real generalization improvement only if validation macro CE drops by at least
`0.10`, Brier drops by at least `0.03`, Pearson reaches `0.20`, and outcome means are ordered with
both adjacent margins at least `0.03`. This audit cannot authorize continuous learning, generation,
arena, or promotion regardless of its result.
