# UFUK: first capacity seed missed its prospective learning thresholds

Intermediate boundary, not the final four-run integrity/gate report. Both
seed20261030 arms completed their fixed40000updates with normal maximum-update
stops; both selected step34000 by the previously frozen common-panel rule.
Second seed20261102 runs and native replay/checkpoint/MLXCPU audits remain active
or pending. No threshold, seed, learning rate, dataset or budget was changed.

Source f4c10e549c82fa35e7d595ccada90a15dccfd613; actual unchanged controllerV2
87d45d879f499cb3cafdd83b8cb145cd9bf8a260dc8383088f3b7d8fc0083446.
Protocol: [prospective capacity plan](UFUK-broad-capacity-preregistration-20261004.md).
Both start from identical e8 functions with optimizer reset, then native
step1000 fresh-process resume. Full training530036rows; exact same20000heldout
rows/group and within-seed batch streams. These data are teacher-generated,
not new genuine self-play outcome learning.

| Selected34000 metric | Narrow | Wide |
|---|---:|---:|
| Anchor policy CE | 2.506062844848633 | 2.4663724239349367 |
| Anchor value CE | 0.5829250788360835 | 0.5834585616856813 |
| Anchor Q MAE | 0.2817297342091799 | 0.27766872278749943 |
| Broader policy CE | 2.6153490001678468 | 2.5679066940307615 |
| Broader value CE | 0.5597694928407669 | 0.552203191614151 |
| Broader Q MAE | 0.2915477915763855 | 0.27836029757261277 |
| Equal-group total CE | 3.1320532083466652 | 3.084970435632765 |
| Whole run seconds | 1702.5922277189966 | 3869.3309934260033 |

Initial broader PCE2.649421015167236/VCE0.5776448660373688/QMAE0.30931934928894045;
equal-group total3.1814649969264863 **exactly identical across both arms**.
Wide's broader gains PCE0.0815143211 and VCE0.0254416744 miss their registered
minimum0.10/0.06. QMAE gain0.0309590517 exceeds0.03. Selected macro advantage over
narrow0.0470827727 misses0.05. Retention and lower observed loss cannot substitute
for these failed requirements. This seed cannot qualify, even if the second seed
passes; no averaging, post-game checkpoint choice or extra training is allowed.

Selected model SHA256:

- narrow `d51b86dfd624eff907e031aea8a31faf839f53f401c228724416c9205b8ae559`
- wide `13f2bde4fbb24f44209676116dec257a6cfb801e2f2c2305b1a096b852f97a89`

Both run directories and every native checkpoint remain under
`artifacts/ufuk-broad-capacity-train-seed-20261030-{narrow,wide}-20261004`.
Actual whole times include startup, validation, checkpointing and the planned
process exit/relaunch. Short development unit validations used other available
CPU resources during wide training; these times are not presented as isolated
architecture throughput benchmarks. Fixed final inference/root timings require
quiescent conditions. No additional production training or speed benchmark ran
concurrently.

The frozen controller still completes the second seed and all native/parity
checks, then the once-only formal gate evaluator records the complete outcome.
The conditional arena may run only if both seeds qualify; this first result
cannot authorize it. No promotion, Stockfish-level gain, overall Elo or paper
novelty claim. Local checkpoint preservation does not close the known remote
Release binary upload gap.

The next research alternative is fresh prior-anchored online learning; its
[74-test components](UFUK-online-loop-components-verification-20261004.md) are
prepared, but real e8 CLI preflight and prospective controlled strength tests
still precede production. An implemented learning loop is not a stronger model.
