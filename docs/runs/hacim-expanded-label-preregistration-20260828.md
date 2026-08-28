# HACIM expanded high-budget label preregistration (2026-08-28)

## Hypothesis

SIPER is independently stronger than the raw network, while CIPA and KOK both
failed on a 320-position fit side despite policy-only and joint-trunk learners.
The remaining primary hypothesis is sample sparsity and within-game
correlation, not another target heuristic or longer exposure to the same rows.

## Frozen fresh label set

- Source: the existing KOPRU 96-game replay only; no new self-play generation
- Train/validation positions: 2,048 / 256, stratified within their existing
  disjoint replay shards
- Search: clean depth-1 oracle teacher at 512 and 800 simulations
- Independent verifier: deterministic depth 4 over every legal action
- Workers: 24 search / 8 oracle; MLX batch 48; wait 0.25 ms
- Seed: `2026082857`
- Exclusions: every TERAZI, DOKU, OLCEK, BAG, VERI, SINAV, and KANIT identity

The raw-Q dataset retains all existing evidence thresholds. A failed
fixed-cardinality top-two overlap does not authorize a raw-Q learner. The
unchanged uncertainty-label gate may separately authorize conservative SIPER
target construction: drift cutoff 0.03, labelable at least 95%, common support
at least 95%, stable visit mass at least 80%, stable-Q/verifier Spearman at
least 0.35, positive conservative verified-gain interval, harmful at most 10%,
and regret at most 0.10.

If that gate passes, build the unchanged conservative `min(Q512,Q800)` target
with maximum `KL(target || raw)=0.10` and every existing SIPER strength/safety
gate. Failure stops before learning.

## Frozen transfer test

Use one KOK joint-trunk arm with policy-anchor weight 4, WDL-anchor weight 4,
AdamW `2e-4`, target batch 16, replay-anchor batch 64, and seed `2026082858`.
Split the 2,048 target rows by whole game into 80% fit / 20% internal holdout.
Use 5,184 steps, preserving approximately the prior target-example exposure
per labelled position instead of increasing epochs. Use 8,192 distinct
fit-side replay anchors when available, otherwise every available fit-side
position and record the count before training.

The KOK transfer, policy-KL, WDL-KL, expected-score, and gradient gates remain
unchanged. Passing authorizes only a separately preregistered, fully fresh
external teacher/target and tactical/WDL/calibration/search qualification.
Arena, generation, and promotion remain blocked.
