# DEVRIYE outcome-balanced rolling replay preregistration

## Prior evidence

`devriye-continuous-pilot-20260830-08` accepted updates 1 and 2, then rolled
update 3 back. The final rolling fresh-value window contained only 25 draw rows
versus 321 decisive rows. Natural game-balanced sampling let that short-run class
mix move WDL micro CE from 0.92481 to 0.95005 at the earliest checkpoint, despite
improving macro CE, Pearson, continuation ranking, tactical solves, and mini-arena
score. Existing gates remain unchanged.

## Frozen hypothesis and change

- Hypothesis: the failure is small-window outcome-mixture variance, not a value
  representation collapse.
- Keep Full Gumbel search, stratified continuation starts, replay size, rolling
  window, historical/fresh 32/32 batch allocation, learning rate, update length,
  checkpoint cadence, and all qualification gates unchanged.
- Within the 32-row fresh half only, sample outcome classes uniformly, then games
  uniformly within each class. Historical rows retain their existing mixed
  outcome/natural sampler.
- Require the rolling fresh window to contain win, draw, and loss records before
  training. Missing coverage stops the update; no class is synthesized and no
  max-ply row becomes a draw.
- Use fresh pilot seed `2026083501`. Do not reuse the `-08` generated games or
  change the sampler after observing results.

## Decision

The preregistered three-update chain gates remain authoritative. Passing does not
promote a release champion; it only authorizes continuous generation/promotion
architecture integration followed by its own qualification.
