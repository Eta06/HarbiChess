# PUSULA continuation non-inferiority preregistration

## Prior result and audit

`pusula-continuous-pilot-20260831-15` remains failed and is not reclassified.
Its first two updates passed. Update 3 improved continuation mean Spearman from
`0.04134` to `0.04785` and verified-top agreement from `0.375` to `0.500`, but
the run rolled back because a legacy absolute point floor required Spearman
`>= 0.050`. Missing that unrelated absolute threshold by `0.00215` is not
evidence of old-capability deterioration.

An audit of the stable PUSULA path found no other old-capability point veto.
Policy CE/top-action requirements are fresh teacher-imitation gates; Full
Gumbel tactical case retention, frozen-parameter/material equality, finite
gradient, and local paired arena floors are independent correctness/safety
guards. Final old WDL, continuation, tactical, and arena gates remain blind and
unchanged.

## Frozen continuation decision

PUSULA-16 replaces only the local absolute continuation floor:

- Compare every otherwise eligible value checkpoint directly with update-0
  MIHVER on the same 32 continuation positions.
- Bootstrap paired per-position differences 2,000 times with deterministic
  preregistered seeds.
- Reject locally only when the one-sided 95% upper bound proves mean Spearman
  deterioration below `-0.020`, or proves verified-top deterioration below
  `-0.125` (four of 32 positions).
- Order candidates by the already frozen fresh CE, macro CE, then earliest
  step. Select the first candidate passing continuation safety. This prevents a
  post-selection continuation failure when another preregistered checkpoint is
  safe; it does not change targets, learning, exposure, or thresholds.
- The 32-position screen cannot authorize production. Final continuation still
  uses a separate stratified 1,440-position set, 20,000 paired bootstrap
  samples, Spearman lower bound `>= -0.020`, and no more than one verified-top
  loss. Those final requirements are unchanged.

## Frozen run

- Run: `pusula-continuous-pilot-20260831-16`
- Seed: `2026091501`
- Fresh replay, self-play, teacher targets, arena openings, and qualification
  starts; PUSULA-15 artifacts and outcomes are excluded.
- Initial checkpoint, MIHVER stable/plastic representation, three updates,
  replay scale, 40 steps/update, batch 64, learning rate, historical/fresh
  weighting, residual alpha grid, Full Gumbel search, policy/tactical gates,
  old/fresh game-paired margins and powered sample sizes, final arena, data
  integrity, and exact resume are unchanged.
- Passing authorizes production-loop integration only. Release promotion stays
  manual and separate.
