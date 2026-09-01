# DENGE fresh cumulative pilot preregistration

Date: 2026-09-01

## Authorization boundary

DENGE-2 is a passing diagnostic-only mechanism test. PUSULA-16 and DENGE-1
remain failed. This new run is the first experiment allowed to provide evidence
for production readiness; it cannot auto-promote a checkpoint.

## Frozen run identity

- Run ID: `denge-continuous-pilot-20260901-01`
- Seed: `2026091701`
- Qualified MIHVER SHA-256:
  `6c535585e952b3d8ba5ff2331fe4dc992d94c586fdfa6a7c3c0cbc9e2d229dbb`
- DENGE integration code parent: `00c43fb`
- Three rolling latest-network updates, 40 learner steps per update.
- 768 fresh self-play attempts and at least 192 known-outcome games per update.
- Full Gumbel 256 teacher targets; Full Gumbel 64 self-play search.
- Historical/fresh value batch weight remains 2:1; policy exposure, network
  capacity, learning rate, search allocation, and replay window are unchanged.

Each update splits known fresh games into deterministic, outcome-stratified,
game-disjoint gradient-fit, scalar-calibration, and tuning partitions. The one
positive scalar is selected for fresh calibration CE and clipped by a separate
historical tuning Pearson margin of 0.005. Calibration changes no policy or
representation parameter.

## Frozen local gates

All existing policy imitation, paired old/fresh WDL safety, calibration,
continuation, Full Gumbel tactical, material invariance, gradient, local arena,
artifact hash, and exact-resume gates remain enabled. A failed update rolls back
and fails the chain.

## Frozen cumulative gates

- Final paired arena: 64 games at 64 simulations; score at least 0.50 and lower
  bound at least 0.45.
- Continuation: 1,440 independent positions; mean Spearman deterioration at
  most 0.020 and verified-top loss at most one position.
- Old qualification: 1,536 attempts, at least 384 known games.
- Fresh qualification: 2,688 attempts, at least 744 known games.
- Statistics: 20,000 game-paired bootstrap resamples, 95% intervals.
- Old non-inferiority: CE/Brier 0.003, macro CE 0.005, Pearson 0.010, ECE 0.010,
  absolute ECE at most 0.120.
- Fresh improvement: CE at least 0.002; macro CE, Brier, and Pearson lower
  confidence bounds non-negative; ECE deterioration at most 0.020 and absolute
  ECE at most 0.150.

No threshold, sample count, seed, exposure, or training duration will be
changed after results are observed. Production orchestration remains disabled
unless every gate passes.
