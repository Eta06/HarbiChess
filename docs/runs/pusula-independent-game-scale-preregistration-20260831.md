# PUSULA independent-game scale preregistration

## Failed predecessor

`pusula-continuous-pilot-20260831-10` remains failed and production remains
closed. Its stricter retention gate correctly rejected update 1: every one of
the 360 step/alpha candidates worsened fresh tuning CE and Brier. The least
harmful candidate differed from baseline by only `+0.00000068` CE because alpha
was `0.0078125`; treating that numerical near-no-op as learning would be invalid.

## Bottleneck evidence

The update generated 192 phase-balanced attempts, but only 65 reached a known
terminal result. After the game-disjoint split, value learning had 52 independent
fit games and 13 tuning games. The 2,609 fit positions are not independent value
labels: all positions from one trajectory share one terminal outcome. Thus the
effective value sample size is games, not rows. The observed fresh tuning result
is consistent with a learner that fits correlated trajectory labels without
generalizing to unseen games.

This is not a request for blindly longer training. Learner steps, batch size,
learning rate, target exposure per step, architecture, search budgets, and all
old/fresh margins remain unchanged. Only the number of independent outcome
trajectories feeding the fixed learner budget changes.

## Frozen PUSULA-11 protocol

The replacement run is `pusula-continuous-pilot-20260831-11`, seed `2026091001`.
Before execution, each update is fixed to:

- 768 phase-balanced latest-network attempts: 256 opening, 256 middlegame, and
  256 endgame starts;
- at least 192 known terminal games, otherwise the update fails before training;
- the same game-disjoint 80/20 fit/tuning split, 40 learner steps, batch 64,
  fixed 32 MIHVER-distillation plus 32 fresh-outcome value examples per step;
- fresh tuning CE improvement of at least `0.002`, with no macro CE, Brier, or
  Pearson regression and no ECE deterioration beyond `0.020` versus update-0;
- the PUSULA-10 combined point-margin plus paired-bootstrap historical retention
  gate and the unchanged residual alpha grid.

The 768-attempt count is a threefold independent-game scale test, not a training
duration sweep. At PUSULA-10's known-terminal rate it is expected to yield about
260 known games; the preregistered 192-game floor leaves termination-rate
headroom while providing roughly 154 fit and 38 tuning games.

If any update lacks 192 known games or cannot produce a residual that meets all
local old/fresh rules, the run stops and no final qualification begins. If all
three updates pass, the already preregistered 1,440-position continuation test,
64-game arena, 2,688-attempt/744-known fresh qualification, and unchanged
20,000-sample cumulative gates apply. PUSULA-10 data is not used for fitting,
selection, or qualification. A pass can authorize production-loop integration;
release promotion remains separate.
