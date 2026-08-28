# KOPRU replay and learner-transfer preregistration

This document freezes the KOPRU experiment before any result is observed. The rejected
OMURGA candidate is not loaded, changed, or evaluated. Continuation/repetition transforms,
value-policy reweighting, and root-halving remain disabled.

## Stage A: fresh replay generation

- Run ID: `kopru-fresh-replay-20260828-01`
- Source model: `artifacts/runs/kilic-control-20260826-01/baseline/model.safetensors`
- Required model SHA-256: `5a094285413d52e663441926afe09aa6efda3aa5e5ec195919b54c159d830eca`
- Teacher qualification: `artifacts/diagnostics/omurga-depth1-teacher-20260827-01/diagnostics.json`
- Games: 96, fresh deterministic seeds from run seed `2026082801`
- Actors: 24; depth-1 oracle processes: 8
- Search: 64 simulations, depth-1 deterministic tactical/material bootstrap value
- Maximum game length: 256 plies; exploration: first 30 plies
- Split: deterministic 75% train / 25% validation
- Training: disabled; no checkpoint or candidate may be written

The replay gate is frozen to the code defaults:

- at least 8,000 samples and 75% unique positions;
- opening at least 5%, middlegame 25%, endgame 15%;
- tactical states at least 8%, quiet states at least 35%;
- teacher-value winning/drawing/losing buckets at least 5% each;
- known-outcome winning/drawing/losing buckets at least 3% each;
- at least 12 material signatures and 24 position-structure signatures;
- clean raw-policy telemetry on at least 99% of samples;
- at least 100 comparable raw/teacher top-action Q deltas, at least 55% positive,
  and strictly positive mean delta.

The per-sample Q delta is telemetry, not independent proof that the teacher is stronger.

## Stage B: fresh independent teacher audit

This stage runs only if Stage A passes. It uses 48 stratified positions from the new validation
shard, seed `2026082802`, the same 64-simulation teacher, oracle depth 1, verifier depth 4,
and 2,000 bootstrap samples. The verified action-value delta 95% interval must have a positive
lower bound. Tactical aggregate solve count must not regress and the diagnostic must report the
64-simulation oracle teacher qualified. Failure blocks the learner.

## Stage C: controlled learner transfer

This stage runs only if Stages A and B pass. It uses only the new KOPRU train shard, keeps the
new validation shard isolated, and uses learning rate 0.0002, batch size 64, at most 200 steps,
validation every 10 steps, and early-stopping patience 12. Training duration or exposure will
not be increased after seeing results.

Before any arena is allowed, one frozen checkpoint must pass all of these independent gates:

- teacher-policy imitation validation cross-entropy improves by at least 2%;
- known-outcome WDL validation cross-entropy is no worse than 2% above baseline;
- WDL expected-score ECE is no worse than baseline by more than 0.02;
- raw-policy tactical solve count does not regress;
- 64- and 512-simulation tactical solve counts do not regress;
- no NaN/Inf and maximum gradient norm remains within the existing pilot safety limit.

If the teacher and replay pass but no learner checkpoint passes all transfer gates, no arena,
promotion, or new generation starts. The next investigation moves to loss balance, value/WDL
representation, optimizer schedule, and model capacity rather than changing teacher heuristics.
