# UFUK: joint trunk/policy/value adaptation result

Both model-improvement hypotheses and the context architecture comparison
**FAILED**. No arena, promotion or result-dependent training extension.
The main strong/fast/reliable self-learning and Stockfish/AlphaZero-level goal
remains unmet. [Prospective protocol](UFUK-joint-context-preregistration-20261004.md).
Previous frozen-context, sparse-value and all-legal failures stay failed.

Clean pinned source `099eb9d1cad79f126db8d7d04640688c57a1796b`. Original18f and retained identity
context e146initial transfer start fresh optimizers. Jointly train every parameter
used by trunk/policy/value; only historical21unused material auxiliary frozen.
Registered72497/78993, active72476/78972. Candidate adds6496ordinary attention
parameters, not a novel algorithm. Initial18real-history Torch full/legal-masked
policy+WDL error0; realMLXCPU policy4.291534e-6/WDL9.536743e-7. This is a new
joint experiment, not resuming failed frozen-branch training or optimizer migration.
Original model/replay/old native checkpoints and transfer bytes remain unchanged.

Native80511datasetSHA b34a3870b494b4cc0fe8a1df87960fe5e2406a5c9a2ed341d1b93fed0e0e094f,
48train16validation families/caps40000/20000/seed20261024, unchanged soft teacher
WDL/policy/loss/sampler. CPU1FP32/AdamW1e-4/wd1e-4/clip5/B64/max30000/eval1000/
patience7500/3600whole-wall seconds per arm including repeated input preparation
and1000pause/fresh-process full native model+optimizer+RNG+input+cursor resume.
Total18000real updates/1152000sampled rows; native supervised learning, not self-play.

| Arm | Last update | Selected minimum totalCE | PolicyCE | ValueCE | QMAE | Charged whole wall seconds |
| --- | --- | --- | --- | --- | --- | --- |
|control|7500|0|2.631909264|0.654059799|0.336921674|532.836100|
|context|10500|5000|2.611201321|0.671440148|0.331775062|989.819788|

Original common initial policyCE2.631909264/valueCE0.654059799/QMAE0.336921674.
Control selected0: no gain. Context policyCE gain
0.020707943 <required0.03;
valueCE gain -0.017380349 <required0.02
(negative is worse); QMAE drift
-0.005146612 <=0.01 passes only that subgate.
Architecture vs joint control policygap
0.020707943 <0.03 and valueCEdrift
0.017380349 >0.01: FAILED, no attention
benefit claim. Both arms stop under the ORIGINAL7500patience rule. Core tracks
progress only when totalCE decreases>=1e-4; candidate's patience anchor remained
3000, so stopped10500. Registered strict minimum measured totalCE selection chose
5000 (slightly lower); these two rules are different and were not changed after
measurement. No extra game-selected checkpoint or threshold substitution.

All21full native checkpoints (9control/12context) actually loaded with complete
model/optimizer tensors/input hashes/RNG/cursor and matching run profiles. Frozen
auxiliary bitwise exact throughout; stem/value/pairwise/context parameters really
changed. Missing Adam states belong only to the frozen auxiliary. Eight common
RNG checkpoints0..7000 exactly matched reconstructed seed24sample indices.
Actual fresh-process resume at1000completed in both production arms. Before
updates547tests passed/zero skips45.07s; targeted3real joint/native/corruption
tests2.42s/Ruff passed. MLXCPU full optimizer/sampler/next-update exact resume and
Torch full optimizer/RNG/next-update exact resume separately tested. Backend
AdamW defaults differ; no cross-backend optimizer or numerical update equality.

Both SELECTED TRAINED portable models were executed on18full-history probes in
actual Torch/MLXCPU, atol/rtol2e-5: `{"context": {"full_policy": 5.7220458984375e-06, "legal_masked_max": 4.351139068603516e-06, "wdl": 1.9073486328125e-06}, "control": {"full_policy": 4.291534423828125e-06, "legal_masked_max": 3.2186508178710938e-06, "wdl": 9.5367431640625e-07}}`.
AppleMetal/CUDA unavailable, untested; no skipped-device tests hide this limitation.
Full-policy+value eagerlegalmaskedB1CPU1/10warmup/200rotating-order rounds,
3600calls per model, no concurrent training. Control/original median ratio
0.992390257, context/original
**1.409292494**; <=1.5cost gates passed.
Control parameters match original, small timing difference is noise, not a model
speed improvement. Setup/warmup0.553649s
excluded, measurement outer9.306091s.
Encoding/search/game/startup excluded; cost pass cannot reverse failed learning.

NEW24uniformlegal12ply stress families seed24 froze before updates,
bookSHAa570f2e543e3820d300571df6a18569399647b975d83ff6e5e402ded804d17f8,
outside1208source receipts/193priorfirst4histories/30priorarena files. No quality
filter; not natural-opening Elo or complete inherited-pretraining holdout.
Selected controlSHA`186ceab92249a2131489f529eecd2f71e167db91a2e3edd89abf1956b5d20915`;
contextSHA`e0bce0dec6b3dff233c45c95290f17406c5a2bc03471c2ac8cba054091849b7f`.

Existing4CPUquota/16GiB/noGPU/no new paid resources. Author+committer
Emir Tunahan Alim <emrtnhalim@gmail.com>, pushEta06/admin, no coauthors/one file
per commit. Immutable source6files actually downloaded/hash checked. Space19
important start read back exactly/78untouched blocks verified. New binary Releases
upload400BadContentLength still blocked: GitUTF8 evidence is not remote model
backup. Previous89file sparse/search supplement deliberately excluded ACTIVE
joint artifacts, so a new verified supplement must include these21checkpoints.

Next separately preregistered infrastructure prepares exact packed native inputs
for larger CPU-scale data, preserving full encoding/target/sampling and native
resume. One million dense inputs alone need26.624GB before policy targets, over
16GiB. This is not chess strength evidence or another extension of this stopped
run. Broader public data sources are exploratory; incomplete Lichess FENs cannot
be represented as restored full histories/clocks. Current model strength and
self-learning still require controlled independent real games/closed-loop gains.
