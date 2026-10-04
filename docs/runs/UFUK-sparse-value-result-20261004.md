# UFUK: sparse current-position value result

Native learning qualification **FAILED**. All four preregistered value/Q
comparisons failed. No arena, promotion or extension of this failed run.
The main strong, fast, reliable self-learning and Stockfish/AlphaZero-level
goal remains unmet. [Prospective protocol](UFUK-sparse-value-preregistration-20261004.md).

Clean pinned source `7df8a0cceec129d74434eb82c0f294d3cca96c75`. Optional sparse-value schema1
uses832current-piece-square/full-EP-square features and8metadata features,
128clipped units,64clipped hidden units,3STM W,D,L logits. Policy retains the
original104history encoder. Exact history-sensitive terminals stay in search.
Original72497parameters preserved/registered/frozen in candidate;116099new
trainable parameters,188596registered total. The original active-value control
trains31290parameters with41207frozen. Standard sparse MLP mathematics is not a
novel invention; primary pinned Stockfish NNUE feature/loss sources were read,
not copied. No quantized/incremental accumulator implementation is claimed.

Transfer-v1 SHA`c7ebff584db59a7db913c199773c96d73abdf90f8065f8f9726bb687f5d83a37` deliberately preserves policy
and creates a NEW RANDOM VALUE with optimizer reset. Initial value identity
and full training resume are **false**; initial Torch policy error0 but value
logit difference9.456951141. Old18f checkpoint/replay unchanged. Full native
model/optimizer/RNG/input-checksum/cursor resume at500 was actually executed
in fresh processes for both arms; portable weights alone are not full resume.

Native80511dataSHA b34a3870b494b4cc0fe8a1df87960fe5e2406a5c9a2ed341d1b93fed0e0e094f,
48train/16validation whole-family split, caps40000/20000, seed20261022.
CPU1FP32/AdamW3e-4/wd1e-4/clip5/B64/max12000/eval500/patience2500,
1800whole-wall seconds per arm including repeated input setup and pause/resume.
Original soft teacher WDL labels and losses/sampling unchanged. This is native
supervised learning, not self-learning. Minimum valid valueCE selection; fixed
policy means totalCE ordering is equivalent. Random initial candidate valueCE
1.100072824 is NOT the qualification reference. Original control initial
valueCE0.653881637/QMAE0.336808679 is the reference.

| Arm | Last update | Selected update | ValueCE | QMAE | Charged whole wall seconds |
| --- | --- | --- | --- | --- | --- |
|control|2500|0|0.653881637|0.336808679|314.660487|
|sparse|4000|1500|0.877539920|0.431018913|318.524171|

Candidate gain vs original `-0.223658283` versus
required+0.05; gap vs selected control
`-0.223658283` versus+0.03. QMAE gains
`-0.094210233`/+0.02 and
`-0.094210233`/+0.01. Negative gains are worse.
Control stopped2500 with best0; candidate stopped4000 with best1500 under
the original2500patience rule. Total6500real updates/416000sampled rows.
Final candidate last-batch training valueCE0.331353307 versus whole-validation
0.969739925 shows a measured generalization gap; it does not identify its cause.
Failed representation/data/budget combination does not rule out every sparse
value architecture. No checkpoint or threshold was changed after measurement.

All15native checkpoints (control6/candidate9) actually loaded with complete
optimizer entries/input hashes/RNG/cursor. Every frozen parameter bitwise exact;
candidate feature weights actually changed. Common6sampling RNG checkpoints
0..2500 matched seed22reconstructed indices. Missing Adam states only belong to
frozen parameters. PolicyCE2.632765688/top16coverage0.9003 exact at every eval.

Before production **538passed/zero skips/44.86s**, targeted11passed2.53s and
final Ruff passed. Real Torch/MLXCPU loss/gradient/update/portable tests and
within-backend exact next-update fullresume tests passed. Selected trained
models were also executed on18real full-history probes in actual MLXCPU:
`{"control": {"full_policy": 4.291534423828125e-06, "legal_masked_max": 3.2186508178710938e-06, "wdl": 9.5367431640625e-07}, "sparse": {"full_policy": 4.291534423828125e-06, "legal_masked_max": 3.2186508178710938e-06, "wdl": 9.5367431640625e-07}}`, atol/rtol2e-5. AppleMetal/CUDA
hardware unavailable, untested. Torch/MLX AdamW defaults differ; identical
cross-backend optimizer updates/full optimizer migration are not claimed.

Matched full-policy+value legal-masked B1 eager inference on18prefrozen history
probes,CPU1,10warmup/200alternating rounds/3600calls per arm:
median candidate/control **0.684251**,
<=1.05cost gate **PASS**. Setup/warmup
0.273083s excluded; outer measured
wall4.600433s. No concurrent training.
Encoding/search/game/startup excluded: roughly31.6%lower warmed inference
latency cannot override the failed learning gate or establish chess strength.

New24uniformlegal12ply stage stress families seed22 froze before updates,
bookSHA27a3b8a2ee2e639f493b326c652ed62386f8b3c488f1505766dee5963d796ad4,
outside1207source receipts/169prior first4histories/30prior arena files.
Not natural-opening Elo or exhaustive inherited-pretraining holdout.
Selected controlSHA`be950e3a5fa37c964eedf636043ae9ac8a8091062d5409223953f9a5d5132bc4`;
candidateSHA`15882e0e9c967f6231aefa0a7121cef8daedd65895b9ee0cb916ac8cab4c93ed`.

Existing4CPUquota/16GiB/noGPU, no paid resources. Author/committer
Emir Tunahan Alim <emrtnhalim@gmail.com>, push accountEta06/admin, no coauthors,
one meaningful file per commit. Pushed source14files (preregistration+13source)
actually downloaded and hashed. Important start log Space17 read back exactly,
66untouched prior blocks verified. Prior completed context/tactical Space16
and156remote restored UTF8 proofs are preserved with this evidence. Binary
models/optimizers/data remain local; exact Git UTF8 evidence is not binary
backup. GitHub Releases upload400BadContentLength remains unresolved; prior
archives predate these new15checkpoints, requiring another verified supplement.

Next hypothesis will be separately preregistered, not a lucky extension of
this failed training. Current policy may prune good moves; an all-legal bounded
search/control can test that separately from model learning. Strong self-learning
still requires independent matched games and reproducible closed-loop gains.
Laya/decision-model/LLM datasets remain later research, not mandatory integration.
