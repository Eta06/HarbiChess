# OMURGA value/search teacher recovery — 2026-08-28

## Decision

The original neural search teacher remains rejected. A deterministic depth-1
tactical/material leaf evaluator, using the champion policy prior unchanged,
qualified at every frozen search budget. It is accepted only as a temporary
bootstrap teacher. The continuous learner, promotion, and a new large generation
remain blocked because the first fresh learner candidate regressed the tactical
safety suite.

Continuation and repetition target transforms remain opt-in and were disabled in
all runs in this report.

## Frozen causal diagnostic

Artifact:
`artifacts/diagnostics/omurga-depth1-teacher-20260827-01/diagnostics.json`

Controls held constant: model, 32 stratified positions, policy priors, search
configuration, random seed, and budgets. Only leaf value changed.

| Budget | Neural verified delta (95% interval) | Depth-1 oracle delta (95% interval) | Neural qualified | Oracle qualified |
| ---: | ---: | ---: | :---: | :---: |
| 64 | +0.0424 (-0.0032, +0.1110) | +0.0622 (+0.0336, +0.0975) | no | yes |
| 128 | +0.0415 (-0.0046, +0.1060) | +0.0665 (+0.0363, +0.1002) | no | yes |
| 256 | +0.0438 (-0.0087, +0.1158) | +0.0773 (+0.0450, +0.1121) | no | yes |
| 512 | +0.0686 (+0.0158, +0.1421) | +0.1026 (+0.0530, +0.1728) | yes | yes |
| 800 | +0.0502 (-0.0006, +0.1186) | +0.1048 (+0.0543, +0.1733) | no | yes |

The oracle tactical solve counts were 6/8, 7/8, 7/8, 8/8, and 8/8 from
64 through 800 simulations. Counts were monotonic and had no regression. Depth 1
matched the useful depth-2 tactical result while reducing the full diagnostic from
675.6 seconds to 264.5 seconds.

## Value pipeline audit

Artifacts:

- `artifacts/diagnostics/omurga-value-pipeline-20260826-01/diagnostics.json`
- `artifacts/diagnostics/omurga-value-bootstrap-20260827-01/bootstrap.json`
- `artifacts/diagnostics/omurga-value-bootstrap-teacher-20260827-01/diagnostics.json`

The champion value head was effectively untrained: validation expected-value
standard deviation was 0.0016 and WDL cross-entropy was 1.0984, approximately the
uniform three-class baseline. Historical combined-loss selection hid value
regression: the lowest combined-loss checkpoint had value cross-entropy 1.5996,
while the baseline was the best value checkpoint.

The legacy validation shard used target schema 3. Four max-ply games (1,024
positions) were identified and excluded under the current schema-10 semantics.
All current max-ply samples use an unknown/masked value target.

A frozen-trunk, frozen-policy, value-head-only diagnostic reduced validation WDL
cross-entropy from 1.0978 to 0.6967 at learning rate 0.0005 without changing policy
loss. This proved that encoding and outcome perspective were learnable, but the
resulting neural search qualified only at 512 and 800 simulations and was not
tactically monotonic. Outcome calibration alone therefore did not reproduce the
local forcing signal supplied by the oracle.

Static-material distillation and joint outcome/policy/oracle hybrid experiments
were rejected: they improved their optimization metrics but failed to retain the
frozen tactical results. These failures are evidence against hiding the problem by
changing loss weights.

## Fresh schema-10 sanity runs

### Run 01

`artifacts/runs/omurga-qualified-sanity-20260827-01/result.json`

- 24 games, 64 simulations, max ply 128, serial depth-1 oracle.
- 23/24 games hit max ply; validation had zero known value targets.
- Correctly failed terminal-distribution guards.
- Exposed a second guard gap: zero known targets previously appeared as value
  loss 0.0. The learner now rejects this case.
- Self-play: 384.0 seconds; 192,301 inference positions (500.8/s).

### Run 02

`artifacts/runs/omurga-qualified-sanity-20260827-02/result.json`

- 24 games, 64 simulations, max ply 256, eight oracle processes.
- 10 checkmates, one insufficient-material draw, four threefold draws, nine
  max-ply truncations; 4 white wins and 6 black wins.
- Target schema 10; 1,002 train and 1,437 validation positions had known value
  targets.
- Policy validation loss improved from about 8.451 to 7.074 at selected step 70.
- Value validation loss moved from 1.0991 to 1.1016 (+0.23%, inside the frozen
  5% safety band). The later step-190 value loss rose to 3.1269, so early stopping
  restored step 70 as intended.
- Self-play: 494.7 seconds; 306,683 inference positions (620.0/s), a 23.8%
  normalized throughput gain over the serial run despite twice the ply limit.

Post-training tactical checks rejected every saved candidate checkpoint. At step
70, neural search solved only 4/8, 5/8, 5/8, 5/8, and 5/8; the oracle-backed arm
solved 5/8, 6/8, 6/8, 7/8, and 7/8. The baseline oracle-backed teacher solved
6/8, 7/8, 7/8, 8/8, and 8/8. No candidate was promoted or sent to an arena.

## Implemented safeguards

- Deterministic leaf-value oracle and frozen neural/oracle comparison.
- Calibrated material scale and model override hashing.
- Replay/value calibration diagnostic with legacy max-ply detection.
- Value-head-only bootstrap diagnostic; it cannot authorize generation.
- Checkpoint selection now requires value loss to remain within a frozen safety
  ratio; policy improvement can no longer hide value collapse.
- Zero known-value train or validation samples fail the learner pilot.
- Default joint learning rate reduced from 0.002 to 0.0002 and exposed in CLI.
- Qualified depth-1 teacher is opt-in; default learning remains heuristic-free.
- Oracle board reconstruction replaced by equivalent push/pop evaluation
  (`max_abs_delta=0`, 14.5% oracle microbenchmark speedup).
- Optional process oracle removes single-GIL serialization.
- Candidate tactical-retention gate compares baseline and candidate before an
  arena or promotion.
- Dashboard preserves frozen teacher evidence across runs and currently reports
  teacher passed, candidate failed, promotion false.

## Literature alignment

AlphaZero trains policy targets from MCTS and value targets from final outcomes;
the published chess setup used 800 simulations and vastly more games/training
steps than HarbiChess's pilots. The comparison supports treating the current
candidate failure as a data-scale and value-signal problem, not as evidence for
more repetition heuristics. See the
[AlphaZero paper](https://arxiv.org/abs/1712.01815) and the
[accepted general AlphaZero manuscript](https://discovery.ucl.ac.uk/id/eprint/10069050/).

KataGo reports large sample-efficiency gains from auxiliary targets and other
search/training improvements, but the failed HarbiChess hybrid ablations show that
an auxiliary term must be validated rather than assumed helpful. See
[Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) and the
[KataGo methods documentation](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md).

Work on targeted AlphaZero search control explicitly identifies accurate values
inside the search tree and diverse starting states as important for sample
efficiency. This supports the next experiment using more independent, varied
teacher-qualified trajectories rather than repeatedly optimizing 24 games. See
[Targeted Search Control in AlphaZero](https://arxiv.org/abs/2302.12359).

## Next experiment

Do not tune the rejected 24-game candidate. Pre-register a generation-only,
teacher-qualified replay collection with substantially more independent games and
stratified/varied starts. Train only after replay coverage and known-value counts
clear frozen minimums. Compare multiple checkpoints with the value and tactical
guards before any arena. If a larger replay still regresses the teacher, increase
network capacity or adopt a genuinely separate auxiliary value head; do not blend
outcome and tactical semantics into one scalar ad hoc.
