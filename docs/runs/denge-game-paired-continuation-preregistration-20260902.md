# DENGE game-paired continuation preregistration

Date: 2026-09-02

## Prior result and authorization boundary

`denge-continuous-pilot-20260901-01` remains failed. It passed every
preregistered old/fresh cumulative value endpoint, all three local updates,
checkpoint resume, tactical retention, and the final arena, but lost 16 net
verified-top hits on the 1,440-position continuation set. The existing rule
allowed only one lost position, so the run is not reclassified and cannot
authorize production.

The audit found that those 1,440 positions came from only 48 historical game
clusters. Treating the point count as the effective sample size was therefore
incorrect. This replacement protocol is frozen before a new run and uses one
position from each independently generated sealed game, so the paired
bootstrap unit and the sampled unit are both the complete game.

## Frozen continuation hypotheses

The final candidate is compared with update-0 MIHVER on exactly the same
records. All differences are candidate minus MIHVER.

| Old capability endpoint | Pass condition |
|---|---|
| continuation Spearman | one-sided 95% paired-bootstrap lower bound `>= -0.020` |
| verified-top agreement | one-sided 95% paired-bootstrap lower bound `>= -0.020` |

Use 20,000 deterministic bootstrap resamples with seed derived from the frozen
run seed. Both endpoints must pass; neither can compensate for the other.
Candidate continuation correlation must remain positive and the existing
release-relative continuation qualification remains enabled.

## Power calculation and fixed sample

The failed DENGE run is used only as a variance-planning pilot. Across its
paired positions, verified-top delta SD was `0.244214` and Spearman delta SD was
`0.205404`. With one-sided alpha `0.05`, power `0.80`, true assumed drift `0`,
non-inferiority boundary `-0.020`, 15% inflation, and rounding to multiples of
24:

- verified-top requires 1,080 independent games;
- Spearman requires 768 independent games.

The fresh experiment therefore fixes 1,440 distinct sealed games and selects
exactly one deterministic stratified position per game. The initial and final
models see byte-identical states and verifier outputs. If fewer than 1,440
distinct games are available, the gate is inconclusive and the run fails; it is
not extended after observation.

## Frozen cumulative learning and retention gates

All DENGE old/fresh game-paired value gates remain unchanged:

- old capability margins: CE `+0.003`, macro CE `+0.005`, Brier `+0.003`,
  Pearson `-0.010`, ECE `+0.010`, absolute ECE at most `0.120`;
- fresh learning: CE lower confidence bound at least `+0.002`; macro CE,
  Brier, and Pearson lower bounds non-negative; ECE deterioration at most
  `0.020`, absolute ECE at most `0.150`;
- 20,000 game-paired bootstrap resamples at 95% confidence;
- 1,536 sealed old-distribution attempts with at least 384 known games and
  2,688 held-out fresh attempts with at least 744 known games.

The final 64-game search arena still requires point score at least `0.50` and
paired lower bound at least `0.45`. Full Gumbel-256 must solve at least 5/8
tactical cases without losing an initially solved case. All artifact hashes,
schema checks, split disjointness, finite-gradient checks, and exact checkpoint
resume checks remain enabled.

## Frozen learning run

- Run ID: `denge-continuous-pilot-20260902-01`.
- Seed: `2026092201`.
- Same qualified MIHVER checkpoint and zero-output plastic residual start.
- Three updates, 40 learner steps per update.
- 768 self-play attempts per update, Full Gumbel-64 self-play and Full
  Gumbel-256 teacher targets.
- Same rolling window, batch composition, learning rate, policy exposure,
  calibration procedure, network capacity, search allocation, and local gates.
- No DENGE-3 replay, qualification game, or checkpoint enters fitting or final
  evidence.

Passing authorizes production-readiness integration work, not automatic release
promotion. Production continuous generation remains disabled until checkpoint
resume, dashboard, data-integrity, and frozen throughput validation also pass.
