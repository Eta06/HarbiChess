# DEVRIYE fresh value replay preregistration

## Root cause from run 04

`devriye-continuous-pilot-20260830-04` accepted all three sequential updates and the final latest
network scored `53.125%` against the immutable MIHVER start (3-11-2; decisive score `60%`). Policy
imitation improved in every update and tactical strength stayed at least 5/8.

The chain still failed, correctly: final macro WDL CE, Pearson, and continuation ranking were slightly
worse than the MIHVER start. DEVRIYE refreshed policy targets from latest-network search but continued
training value exclusively on old KOPRU outcomes. Therefore it was not a complete policy-iteration
loop and provided no new value learning signal.

## Frozen fresh-replay correction

- Keep three updates, policy target counts, Full Gumbel 256 clean teacher, rolling two-generation
  policy window, minimum-sufficient checkpoint selection, optimizer continuation, architecture,
  learner steps, learning rates, and every existing gate.
- Before each update, generate 12 fresh latest-network self-play games using Full Gumbel 64 behavior
  search, top 16 actions, Gumbel scale 1.0, temperature 1 through ply 30, then deterministic choice,
  and a 96-ply limit.
- Max-ply samples retain unknown value targets; they are never relabeled as draws.
- Require at least four fresh games with known terminal outcomes. Otherwise the update fails before
  learner training.
- Store every generation as a versioned replay shard with latest checkpoint provenance.
- Retain fresh value replay from the latest two generations. Each 64-row WDL minibatch contains 32
  rows from the existing corrected historical sampler and 32 game-balanced rows from rolling fresh
  known-outcome replay. Total WDL batch size and learner steps do not increase.
- Continue measuring WDL on the immutable trajectory-disjoint validation set.

The new run uses fresh seed `2026083401`. Final WDL micro/macro CE and Pearson plus continuation
ranking must meet the unchanged no-regression comparison to MIHVER. All previous policy, material,
tactical, mini-arena, and final-arena gates remain unchanged. Passing authorizes production
continuous generation integration, not direct model promotion.
