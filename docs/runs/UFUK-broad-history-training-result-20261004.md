# UFUK: two-seed broad-data teacher pretraining result

**Registered learning/retention gates PASS separately in both seeds. Actual game
strength, speed and genuine closed-loop self-learning are still unestablished.**
The main Stockfish/AlphaZero-level goal remains unmet; no promotion or novelty.
The original failed hypotheses/data remain preserved.

Hypothesis and registration: [two-seed protocol](UFUK-broad-history-training-preregistration-20261004.md), committed before updates.
Original18f/identical72,497parameter architecture/used trunk+policy+STMsoftWDL,
FP32CPU1/B64/AdamW1e-4/no train cap/max30000/eval1000/patience7500.
Frozen21unused material-readout parameters stayed bitwise unchanged.
Control60375native versus candidate530036merged training rows; exact same common
old/new20000row validation panels PERseed, source-family heldout/overlap guards.
Each run allowed3600WHOLEseconds. Actual control/candidate work differs because
both follow the same registered early stop; this is matched maximum budgets and
stopping rule, not equal actual updates/time or a pure equal-work causal contrast.
Both sources and all target semantics/architecture/search assumptions retained.

| Seed/arm | Updates / selected | Anchor policyCE / WDLCE / QMAE | Broader policyCE / WDLCE / QMAE | MacroCE | Whole seconds |
| --- | --- | --- | --- | --- | --- |
|20261026 initial|0 / 0|2.632999/0.654390/0.337088|2.811066/0.763692/0.409109|3.431073|—|
|20261026 control|9500 / 2000|2.625401/0.652812/0.331188|2.800758/0.762395/0.404312|3.420683|468.173|
|20261026 candidate|30000 / 30000|2.532782/0.604876/0.295946|2.657729/0.577933/0.308066|3.186660|1444.505|
|20261028 initial|0 / 0|2.632770/0.653487/0.336605|2.811727/0.769072/0.411545|3.433528|—|
|20261028 control|9500 / 2000|2.623900/0.661187/0.335403|2.798781/0.778129/0.407631|3.430999|464.186|
|20261028 candidate|30000 / 30000|2.535674/0.614471/0.299697|2.660756/0.585717/0.310803|3.198309|1446.555|

Both candidates improve broader policy>=0.05/value>=0.03/QMAE>=0.015, retain
anchor policy<=+0.03/value<=+0.02/QMAE<=+0.015 and beat their own selected native
control macroCE by>=0.05. Every criterion passed; neither seed substituted for
the other. Validation loss is a point estimate on fixed development families,
not Elo, actual playing strength, imitation beyond teacher or independent data
replication. Both seeds use the same completed teacher corpus, different sampling
and deterministic heldout row subsets. Target entropy is descriptive, not a
proof of attainable model error or an architecture bottleneck.

Seed20261026:broader policy gain0.153337,value gain0.185759,QMAE gain0.101044;macro gain versus control0.234023.

Seed20261028:broader policy gain0.150970,value gain0.183356,QMAE gain0.100741;macro gain versus control0.232690.

Production training79,000updates/5,056,000sampled row presentations. Four
additional isolated resume-audit copies1000updates each/256000presentations are
NOT learned progress. They are separately charged and reproduce real resumed
production step2000 from native step1000:every model/Adam tensor, Torch RNG,
full chained sampled-row index trace and cursor BITWISE. All four PASS. No
weights-only import described as full resume; no cross-framework optimizer
migration. Every84complete production checkpoint actually loaded, raw+packed
input checksums verified, all sampler RNG/index chains reconstructed from seed.

Whole four-run plus resume-audit controller3999.501022s; complete84state audit89.947054s afterimport (whole parent receipt preserved separately). No GPU or new paid resources.

Actual selected four-model18true-history probe Torch/MLXCPU full4672policy/WDL/legal-mask parity PASS, maxabsolute error5.24520874e-06, unchanged2e-5atol/rtol. AppleMetal/CUDA/BF16device UNTESTED. Real Linux MLXCPU inference/gradient/update/native resume support remains; AdamW defaults differ. Current source fullsuite587PASS/zero skips, sourceeb2880767d1671195ec6136031b6f802308f79bb/54.47pytest seconds/58.017whole parent seconds.

Immutable training sourcef4c10e549c82fa35e7d595ccada90a15dccfd613, generation
source8fa4f2907d1a661ef30e995ee025ceaeb1c1a0df, merged raw SHA
ca2912fc37845fe395f0e445be72de805d99ac87d5b7ce2b5d65cfe4c830eee7,
prepared SHA8c9a06de8be2c1f15a44569cd7bfd0fbc234907242f2145dfe8f3dee5a3fb4f5.
No inputs/oldweights/oldhistory changed. Version1 merge and lossless prepared
schema1 are distinct input conversions, not training resume. RuntimeTorch2.14.1+cpu/
CPU1/deterministicTrue and exact full native profiles retained for continuation.

Only PRIMARYseed20261026selected candidate and control may enter the already
registered48newsource-game/96game-per-arm/five-arm strength and speed protocol.
No games have been played at this reporting boundary. No alternate checkpoint,
seed, easier roots or extended budget is selected using game results.

| Selected model | Step | SHA-256 |
| --- | --- | --- |
|20261026/control|2000|6484ccd5ba42811e41d39b52ec41b58f09cffd304442d162e99c498ec5513f6c|
|20261026/candidate|30000|e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03|
|20261028/candidate|30000|a6e7974d997761614bb1efb0fb2b6ea501cb61b3d2f21730bd396764fcdfdb3d|
|20261028/control|2000|a4544327acdf3c7c1117f1a373d9f1640f23be84647467d66fad530cdeacd6fd|

[Exact result evidence](UFUK-broad-history-training-result-evidence-20261004.json)
contains commands, whole-clock receipts, failed/successful technical guards,
source/model/data hashes, all interval traces, complete native audits, exact
prospective scripts and Space readback preservation. Large models/full Adam/RNG/
packed arrays remain local verified binary archives. Authorized GitHub Releases
upload400BadContentLength remains unresolved; new remote binary backup is
INCOMPLETE, not fixed by text in Git. No automatic model promotion.

[Seventh model archive inventory](UFUK-artifact-broad-training-supplement-20261004.json):351new files/355696047raw bytes/147054070tar bytes SHA
ec6c3ced7108c6bc0cd14ffa0ce0e271c29fcb77405afe1bf6a4b40005b84e87;
353stream-restored/hash-verified members,15.642seconds. Includes all84native
checkpoints and terminal training/check traces; all six prior archive bytes
and prior immutable artifacts verified unchanged. Remote upload still incomplete.
