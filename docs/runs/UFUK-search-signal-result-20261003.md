# UFUK own-search signal: bottleneck hypothesis not met

All96 frozen positions completed under the registered900s limit, no missing
cases. Residual value-oracle regret gain0.017792 did not meet0.05 threshold;
128-vs16 gain0.011943 also failed and its interval includes zero. Do not keep
claiming the old severe value collapse is the sole current bottleneck.

Source clean detached`08b5cac6488f165a74f01f14dbcb50e1cd65123e`, unchanged
`evaluation.teacher_probe`; [preregistration](UFUK-search-signal-preregistration-20261003.md).
BaselineUFUK18f2aae5…;latest failed-self-playd5f5bbc5…; exact hashes in metadata.
Panel SHA`e2ae34e6773392d5ff8b143b2b8b081ad0eeaf0be87b65e034ae7b9e48545c23`
matches AYNA exactly: same eight old development families,96 full histories from
MERCEK16-simulation arena. No fresh test-set or general chess strength claim.

| Choice arm | Restricted mean expected-WDL regret |
| --- | ---: |
| Baseline raw policy | 0.00616146 |
| Failed self-play raw policy | 0.00493229 |
| Baseline FullGumbel16 | 0.01925000 |
| Baseline FullGumbel128 | 0.00730729 |
| Baseline policy / engine-leaf-value FullGumbel16 | 0.00145833 |

Oracle-vs16 conditional family bootstrap[0.001443,0.042786], effect0.017792;
threshold failed despite positive bootstrap lower. Conservative bound
[−0.942531,+0.978114].128-vs16 bootstrap[−0.009229,+0.040542], effect0.011943.
Raw-vs16 difference0.013089 bootstrap[−0.006365,+0.040547], not conclusive.
Latest-raw-vsbaseline-raw difference0.001229[−0.000953,+0.004641], also inconclusive.
The point estimates suggest search can damage already useful policy choices;
they do not prove raw play wins games. Latest search was not measured here.

Root value MAE0.241875 for both models; frozen value identical as expected.
Old AYNA initial MAE0.90697 is on this same panel, but training/data/architecture
changed together; this is not a single-mechanism attribution. Native32k teacher
top16 coverage86/96=0.895833;4k-vs32k bestmove agreement62/96=0.645833.
Reference budget itself is noisy. Restricted regret uses only tested candidates,
not exact minimax; many extreme engine WDLs are saturated. Source-game outcomes
give CE1.310371 on all96 rows, conditional on weak players, not optimal-play truth.

Actual wall64.426164s,parent userCPU43.594158s,RSS255516KiB; engine child CPU
not reported by this collector.2095 engine queries/13,802,452 actual nodes,
15s watchdogs/fresh TT/coherent unbounded packets, Stockfish19/one thread/Hash16MiB,
1024-node leaf oracle/4096+32768 reference/32768 forced candidate roots.
Compute is unequal; no fast oracle deployment claim. Existing CPU allocation only.
Overlapped source-integrity arena end and tests; no isolated wall-time comparison.

Independent audit replayed all96 histories,checked every action/score packet,
frozen panel hash,frozen value equality,normalized WDLs and summed2095 query nodes.
Exact scripts, commands and raw receipts in
[text evidence](UFUK-search-signal-evidence-20261003.json); binary weights remain
local, Release400 still blocks remote artifact backup.

Decision: no repeated failed training. Next isolate raw-policy move selection vs
the same network's16-simulation search in a preregistered colour-paired arena on
new frozen roots. Preserve search as default until that game-level comparison;
do not infer improvement from this small diagnostic or call conventional search
mechanisms a novel research contribution. Laya is optional and not part of this run.
