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

## Fresh value-selection experiment recorded before execution

Run `pusula-continuous-pilot-20260830-03` remains failed because it missed
`fresh_ce_superior` and `old_ece_noninferior`; it is not reclassified and none of
its replay enters the next run. The audit found that the implementation stopped
value validation after the first locally safe gradient step. Consequently each
of its three selected value checkpoints was local step 1, despite the frozen
40-step budget. It also had no game-disjoint fresh tuning partition for choosing
which checkpoint transferred new evidence.

The next experiment changes checkpoint selection, not compute or thresholds:

- deterministically reserve 20% of each update's known fresh games, stratified by
  result, as a game-disjoint fresh tuning partition; these rows never train;
- evaluate all 40 value checkpoints against both historical tuning and rolling
  fresh tuning;
- eligible checkpoints must pass the existing historical local safety bounds and
  improve fresh CE, macro CE, Brier, and Pearson in their correct direction;
- among eligible checkpoints select minimum fresh CE, then minimum fresh macro
  CE, then earliest step; policy checkpoint selection remains unchanged;
- keep every cumulative margin, 768-attempt qualification size, search budget,
  replay attempt count, learner step count, and final guardrail unchanged.

The fresh run uses seed `2026090301`. Run `03` is diagnostic evidence for the
hypothesis only and is excluded from training, checkpoint selection, confidence
intervals, and power calculations.

## Stable-function rehearsal experiment recorded before execution

Run `pusula-continuous-pilot-20260830-04` remains failed and cannot authorize
production. It passed all fresh-learning endpoints by wide margins, but failed
all five old-capability non-inferiority endpoints. The audit identified two
mechanisms that permitted cumulative forgetting despite a frozen MIHVER base:

- each local checkpoint was compared with the immediately preceding update, so
  several locally acceptable changes accumulated beyond the original-model
  margin;
- the historical half of every value batch was trained again against sampled
  game outcomes instead of preserving the qualified MIHVER WDL function. This
  allowed the plastic residual to rewrite calibration on the old distribution
  even though all MIHVER parameters stayed frozen.

The next fresh experiment changes the continual-learning constraint, not the
final statistical gate:

- historical fit examples receive the original MIHVER soft WDL distribution as
  a distillation target; fresh fit examples retain empirical soft outcome
  targets formed only from byte-identical fresh states;
- the fixed 32 historical / 32 fresh value batch composition remains unchanged;
- every value checkpoint is compared with the update-0 MIHVER metrics on the
  historical tuning partition, never merely with the preceding update;
- checkpoint eligibility uses the already frozen point margins `+0.003` CE,
  `+0.005` macro CE, `+0.003` Brier, `-0.010` Pearson, and `+0.010` ECE with
  absolute ECE `<=0.120`, together with all fresh-direction requirements;
- the independent final historical holdout remains invisible until the one-time
  cumulative test.

All final bootstrap margins, 20,000 samples, replay/search/training budgets,
three-update horizon, arena/tactical/continuation safeguards, and exact-resume
requirements remain unchanged. Run `04` replay and checkpoints are excluded.
The replacement uses seed `2026090401` and a new output directory. This is a
fresh prospective test; run `04` is not reclassified.

## Residual trust-region experiment recorded before execution

Run `pusula-continuous-pilot-20260831-05` remains failed. Stable-function
distillation allowed update 1 to pass, but update 2 was rolled back: even its
first full optimizer step put historical tuning CE `0.000272` beyond the frozen
point margin while already improving every fresh metric. No update-2 checkpoint
satisfied both sides of the constraint, so update 3 and final qualification did
not run.

This identifies optimizer-step resolution at the cumulative boundary rather
than lack of a useful fresh gradient. The next experiment adds a deterministic
value-only trust-region line search after the unchanged 40-step learner:

- for every saved value checkpoint, scale its plastic-value parameter delta
  from the accepted update boundary by the fixed alpha grid
  `{1, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125}`;
- policy parameters are not interpolated and retain their independent existing
  checkpoint selection;
- an interpolated value state is eligible only if it satisfies the frozen
  update-0 historical tuning point margins and improves fresh CE, macro CE,
  Brier, and Pearson in their required directions;
- select minimum fresh CE, then minimum fresh macro CE, then earliest learner
  step exactly as before; reset optimizer after headwise composition exactly as
  before;
- alpha zero is excluded because a no-learning state cannot satisfy the fresh
  improvement requirement.

This changes neither gradient direction nor training exposure and introduces no
new final tolerance. All cumulative bootstrap, sample-size, compute, replay,
search, arena, tactical, continuation, data-integrity, and resume gates remain
unchanged. Run `05` data is excluded. The prospective replacement uses seed
`2026090501` and a new output directory; run `05` is not reclassified.
