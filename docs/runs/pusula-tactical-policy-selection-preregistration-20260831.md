# PUSULA tactical policy-selection preregistration

`pusula-continuous-pilot-20260831-12` remains failed. Its first update produced
246 known games and selected a value checkpoint that passed every historical and
fresh numeric rule, but the composed candidate reduced Full Gumbel tactical
solve rate from 5/8 to 4/8 and lost a baseline-solved case. No final arena or
qualification ran and production remains closed.

The audit found a checkpoint-selection ordering bug. Policy checkpoints at
steps 20, 30, and 40 all passed the frozen policy imitation gate, but the runner
always chose the earliest, step 20. Full Gumbel tactical retention was evaluated
only after that choice, so steps 30 and 40 were never tested even though search
tactical capability is a required property of the composed policy/value model.

PUSULA-13 fixes selection without changing search allocation or any gate. For
each policy checkpoint that passes the existing CE and top-action rule, the
runner composes the one already-selected safe value checkpoint and evaluates the
same frozen 8-case Full Gumbel-256 tactical suite. It chooses the earliest policy
checkpoint that retains at least 5/8 and every baseline-solved case. If none do,
the update fails. Tactical outcomes and rejection reasons for every candidate
are stored in result telemetry.

The fresh run is `pusula-continuous-pilot-20260831-13`, seed `2026091201`. It
keeps the 2:1 historical/fresh value weighting, 768 attempts and 192-known floor
per update, 40 steps, all replay/search/optimizer settings, old/fresh margins,
residual grid, 1,440-position continuation, 64-game arena, 1,536-attempt sealed
old qualification with 384-known floor, 2,688-attempt fresh qualification with
744-known floor, and 20,000-sample final bootstraps unchanged. PUSULA-12 data is
not reused. Passing can authorize production-loop integration but cannot promote
a release checkpoint automatically.
