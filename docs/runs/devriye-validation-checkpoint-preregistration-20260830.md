# DEVRIYE validation-checkpoint selection preregistration

## Evidence from the failed chain

After dtype isolation, `devriye-continuous-pilot-20260830-02` accepted update 1. Update 2 improved
fresh teacher policy CE (`2.86823 -> 2.69953`), macro WDL CE (`0.94516 -> 0.94023`), Full Gumbel
tactical (`5/8 -> 6/8`), and mini-arena score (`56.25%`). It rolled back solely because final-step
micro WDL CE moved `0.90539 -> 0.91755`, exceeding the frozen `+0.01` relative limit by `0.00217`.

The runner evaluated only step 40. That is an invalid continuous-learning checkpoint policy: a
bounded stochastic update must retain optimizer-compatible validation checkpoints rather than
blindly accepting its last minibatch.

## Frozen correction

- Keep 40 maximum steps, batch 64, learning rate `1e-4`, loss weights, rolling window, architecture,
  data scale, and every existing gate unchanged.
- Evaluate fresh teacher policy and held-out WDL at steps 10, 20, 30, and 40.
- A checkpoint is numerically eligible only if it passes the unchanged policy-imitation and WDL
  gates against the preceding accepted latest network.
- Before any continuation/tactical/arena result is seen, select the eligible checkpoint with minimum
  fresh teacher policy CE; break ties by macro WDL CE and then the earlier step.
- Restore both weights and optimizer state from that checkpoint. Run downstream gates once on the
  selected checkpoint. No retry among downstream failures.
- If no numeric checkpoint qualifies, roll back to the previous latest state.

Repeat the full three-update pilot on fresh non-overlapping target selections and arena openings with
seed `2026083201`. All original thresholds and sample sizes remain frozen. The failed run and its
targets remain immutable development evidence.
