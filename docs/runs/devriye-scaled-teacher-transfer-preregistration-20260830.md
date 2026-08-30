# DEVRIYE scaled teacher transfer preregistration

## Evidence leading to this pilot

- The policy head has 1,200,772 trainable parameters.
- The prior pilot used only 96 unique Full Gumbel teacher rows per update but
  consumed 2,560 policy examples, or about 26.7 repeated passes in update 1.
- A cached 200-step diagnostic on the failed `-09` rows reduced validation CE
  only through step 80, never recovered teacher top-action agreement, then
  overfit. Longer training is rejected.
- The earlier KOK audit independently found that 320 searched fit positions did
  not transfer to unseen games and identified high-budget label sparsity/game
  correlation rather than capacity or WDL instability.
- Natural fresh WDL sampling was unstable when a short rolling window contained
  almost no draws; uniform W/D/L sampling then over-weighted decisive outcomes
  relative to the fixed validation distribution (53.18% draw, 23.33% loss,
  23.50% win).

## Frozen pilot

- Fresh seed: `2026083601`.
- Three rolling latest-network updates; rolling window remains two generations.
- Per update, generate 768 train and 192 validation Full Gumbel-256 targets from
  distinct games/positions. No target row is reused across updates.
- Keep 40 learner steps, batch 64, learning rate `1e-4`, and validation every 10
  steps. This is about 3.33 policy passes in update 1 and 1.67 passes once the
  two-generation window is full; training duration is not increased.
- Keep 12 phase-balanced continuation games, Full Gumbel-64 behavior, the
  96-additional-ply limit, minimum four known terminal games, and max-ply value
  masking unchanged.
- Keep the historical/fresh WDL batch split at 32/32. The fresh half uses the
  fixed per-batch outcome allocation 8 loss / 16 draw / 8 win, sampling games
  uniformly within each outcome. All three outcomes must exist in the rolling
  window; otherwise stop before learning.
- Keep every policy, WDL, material, continuation, Full Gumbel tactical,
  per-update arena, and final chain gate unchanged. Thresholds cannot change
  after results.

## Decision

Only a full three-update pass authorizes production continuous-loop integration.
It does not directly promote a release champion. A failure triggers another
root-cause audit without increasing training duration or weakening gates.
