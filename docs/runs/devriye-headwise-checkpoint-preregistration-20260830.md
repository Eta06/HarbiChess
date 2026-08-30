# DEVRIYE head-wise checkpoint preregistration

## Evidence

`devriye-continuous-pilot-20260830-11` supplied 34 independent terminal games
and 768 searched policy-fit rows. It failed only because no single validation
instant satisfied both independent heads:

- At step 10, WDL passed (`micro CE 0.91405`) but policy CE improvement was
  0.00601, below the fixed 0.01 requirement.
- At step 20, policy passed (CE 2.89927 to 2.88178; top agreement 10.42% to
  25.00%) but WDL micro CE exceeded its regression allowance.
- Policy and global/invariant WDL parameter sets are disjoint and the shared
  trunk is frozen. A single checkpoint time is therefore an accidental coupling,
  not a joint representation constraint.

## Frozen change

- Fresh seed: `2026083801`.
- Keep 96 continuation games, 24 terminal-game floor, 768/192 Full Gumbel-256
  targets, 40 steps, batch 64, learning rate, replay mix, and all downstream
  gates unchanged.
- Select the earliest validation checkpoint that independently passes all policy
  gates and the earliest checkpoint that independently passes all WDL gates.
- Compose policy parameters from the policy checkpoint and invariant/global WDL
  parameters from the WDL checkpoint. Frozen parameters must remain bitwise
  identical.
- Recompute policy and WDL metrics on the composed model and require the same
  gates again before tactical, continuation, and arena evaluation.
- Adam has a single global bias-correction step, so moments from different
  checkpoint times cannot be safely spliced. After a successful composition,
  reset optimizer moments while preserving composed network weights and the
  monotonic learner-step counter. A later rollback restores the complete
  pre-update weights and optimizer state.
- Record both selected local steps and the optimizer reset in telemetry/result
  provenance.

## Decision

No metric threshold changes. All three composed updates and the existing final
chain gates must pass before continuous production integration. Release champion
promotion remains separate.
