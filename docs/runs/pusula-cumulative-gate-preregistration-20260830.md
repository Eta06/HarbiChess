# PUSULA cumulative non-inferiority gate preregistration

## Scope and separation from old evidence

This protocol applies only to a new fresh rolling-replay experiment. YELKEN and
DEVRIYE remain failed diagnostic runs and will not be reclassified under this
gate.

The design follows three general principles: define a practically meaningful
non-inferiority margin before observations, analyze paired differences, and size
the experiment from the hypothesis and desired power. See the FDA
non-inferiority guidance and the NIST paired-observation confidence-interval
reference:

- https://www.fda.gov/regulatory-information/search-fda-guidance-documents/non-inferiority-clinical-trials
- https://itl.nist.gov/div898/handbook/prc/section3/prc312.htm

## Statistical unit and interval

- The independent bootstrap unit is a complete generated game, never an
  individual position. Every resample retains all positions from the selected
  game, preserving within-game dependence.
- Baseline and candidate predictions are evaluated on the exact same records;
  all statistics use paired differences.
- Use one-sided 95% percentile intervals from 20,000 game-cluster bootstrap
  samples, seed `2026090101`.
- All listed co-primary conditions must pass. This is an intersection-union
  decision: one endpoint cannot compensate for another.
- The final cumulative test is performed once after update 3. Intermediate
  checks are catastrophic safety floors only and cannot authorize production.

## Frozen old-capability non-inferiority margins

Candidate minus original MIHVER on the untouched historical stability set:

The original game-disjoint MIHVER validation partition is reserved exclusively
for this final old-capability test. A second, game-disjoint tuning subset is
split from the historical training partition before update 1 and is used for
checkpoint selection and all intermediate safety checks. The old-capability
holdout therefore cannot influence training, checkpoint choice, or early stop.

| Endpoint | Pass condition |
|---|---|
| position-weighted WDL CE | one-sided upper bound `<= +0.003` |
| equal-class macro WDL CE | one-sided upper bound `<= +0.005` |
| multiclass Brier | one-sided upper bound `<= +0.003` |
| expected-score Pearson | one-sided lower bound `>= -0.010` |
| ECE-10 | one-sided upper bound `<= +0.010` and candidate absolute ECE `<= 0.120` |

These are practical deterioration limits, not values fitted to YELKEN. They are
approximately sub-percent loss changes, retain MIHVER's outcome separation, and
are much smaller than the catastrophic floors used by DEVRIYE.

## Frozen fresh-learning superiority requirements

Original MIHVER minus final candidate on a completely held-out fresh replay set
for loss metrics, and candidate minus MIHVER for Pearson:

| Endpoint | Pass condition |
|---|---|
| WDL CE improvement | one-sided lower bound `>= +0.002` |
| macro WDL CE improvement | one-sided lower bound `>= 0.000` |
| Brier improvement | one-sided lower bound `>= 0.000` |
| expected-score Pearson improvement | one-sided lower bound `>= 0.000` |

Fresh learning must therefore be statistically positive on every WDL view; old
non-inferiority alone is not success.

## Power plan and fixed sample size

Planning artifact:
`artifacts/runs/pusula-cumulative-power-plan-20260830-01/result.json`.

- Planning input is the rejected DEVRIYE update-3 diagnostic checkpoint. It is
  used only to estimate paired game-level variance.
- One-sided alpha `0.05`, power `0.80`, 15% variance/missingness inflation, round
  to multiples of 24.
- Old CE: SD `0.004469`, NI margin `0.003`, assumed true deterioration `0.0`;
  computed 17 inflated games, rounded to 24.
- Fresh CE: SD `0.008571`, required improvement `0.002`, design alternative
  `0.006`; computed 34 inflated games, rounded to 48.
- The preregistered diversity/correlation floor dominates: exactly 192 known
  terminal games are required in the fresh final qualification set.
- Generate exactly 768 phase-stratified qualification attempts. If fewer than
  192 yield known terminal outcomes, the experiment is inconclusive and cannot
  be extended after results.

## Fresh continuous pilot

- Start from the qualified MIHVER checkpoint wrapped by the zero-output
  stable-base/plastic-residual architecture.
- MIHVER value parameters remain frozen. Policy uses the unchanged qualified
  DEVRIYE policy-learning path; no search/temperature/target heuristic changes.
- Run exactly three updates, 192 phase-stratified self-play attempts per update,
  24 workers, 64 self-play simulations, max ply 96.
- Starting positions are distinct replay states across the three updates and are
  never reused. Multiple states may originate from one historical trajectory;
  requiring a distinct historical game per attempt would exceed the fixed pool
  and does not add an independent outcome because every generated continuation
  still receives its own seed and game-cluster identity.
- Retain the existing 768 train / 192 validation Full Gumbel-256 policy targets,
  40 learner steps, batch 64, learning rate `1e-4`, and two-generation rolling
  buffers.
- Merge contradictory byte-identical fit states into empirical soft WDL targets;
  unique states remain one-hot. No validation outcome enters target aggregation.
- The word `validation` in per-update policy targets means the fixed tuning
  subset split from historical training data. It does not refer to the untouched
  final old-capability holdout.
- Per-update local gates retain DEVRIYE policy imitation, WDL catastrophic
  floors, continuation floor, Full Gumbel tactical retention, immutable MIHVER,
  and search score floor.

## Final non-statistical and integrity gates

In addition to the cumulative statistical gate:

- continuation mean Spearman may decline by at most `0.020`, with paired
  bootstrap lower bound at least `-0.020`; verified-top agreement may decline by
  at most one position out of 32;
- Full Gumbel-256 solves at least 5/8 tactical cases and loses no MIHVER-solved
  case;
- 64-game final paired search arena has point score at least `0.50` and paired
  bootstrap lower bound at least `0.45`; this is a production-safety gate, not a
  champion promotion claim;
- replay target schema, source hashes, split disjointness, finite gradients, and
  checkpoint hashes verify;
- every accepted update publishes a checksummed boundary checkpoint containing
  model, optimizer, learner step, rolling replay/target provenance, and the next
  update seed. Sampling streams are deliberately reconstructed from the global
  seed plus update index at boundaries, so no hidden mutable sampler state crosses
  updates. A restored checkpoint's next controlled step must match an
  uninterrupted reference within `1e-7`.

Production continuous generation is authorized only if all three local updates,
the 192-game fresh cumulative gate, continuation/tactical/search safeguards, and
resume-integrity test pass. Promotion remains a separate later decision.

## Replacement-run correction recorded before execution

Run `pusula-continuous-pilot-20260830-02` remains failed and is not evaluated
again. It exposed two code paths that contradicted this protocol: a legacy
absolute macro-CE cutoff from a different validation distribution made the new
tuning baseline fail before learning, and a raw-policy tactical check rejected a
candidate despite the preregistered Full Gumbel `5/8` capability being retained.

The replacement run keeps every cumulative margin, sample size, compute setting,
and non-statistical final gate above unchanged. Its local WDL safety check is the
already specified previous-update-relative catastrophic bound (`+0.01` CE,
`+0.01` macro CE, `-0.02` Pearson, plus the existing outcome-margin and ECE
floors). Its tactical safety check is Full Gumbel `>=5/8` with no loss of a
baseline-solved case; raw-policy solve count remains telemetry, not a gate. The
replacement uses new seed `2026090201`, new continuation trajectories, new
teacher selections, and a new output directory. No result from run `02` enters
qualification or power estimation.
