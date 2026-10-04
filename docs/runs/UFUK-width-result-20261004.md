# UFUK: same-initial-function width result

The registered native learning qualification **FAILED**. No width arena, model
promotion or extra updates. Main strong/fast/reliable self-learning and
Stockfish/AlphaZero-level goal remains unmet; no novel algorithm or Elo claim.
[Original protocol](UFUK-width-preregistration-20261004.md).

Actual clean sourceaa4b28fd87a056d123c365bb4bfb4fdbccc71dba; the original native model18f
was preserved. Weights-only transfer SHAab077bbcfe11f64e8e1918b0e8dd538ae11222d7fef846a3583724406ba8933f,
72497→302801registered parameters, shared16→64 and independent value16→32.
Old active channels stay as exact subblocks, new random features have initially
zero bridges into inherited outputs. Transfer schema1 resets optimizers; it is
not a full optimizer migration. Within each arm250pause→fresh-process full
optimizer/RNG/cursor/input resume completed successfully at the fixed source.

On18frozen full-history probes initial full/masked Torch outputs exactly matched;
real MLXCPU maxpolicy error4.29153442e-06 and WDL error
9.53674316e-07, below2e-5 tolerance. AppleMetal/CUDA unavailable.
Native initial measurements exactly matched both arms: policyCE2.632536949539,
valueCE0.653973489664,QMAE0.336884064344. Seed17/cappedvalidation differs from
the older seed05native observer; older entropy measurements motivate this test
but are not silently claimed as the exact entropy of this new row sample.

Same native80511datasetSHA b34a3870b494b4cc0fe8a1df87960fe5e2406a5c9a2ed341d1b93fed0e0e094f,
40k/20k caps,48training/16validation development families; sameCPU1FP32
AdamW5e-5/wd1e-4/clip5/B64/seed20261017/max6000/eval250/patience1500. Patience
uses unchanged totalCE; experiment selection uses minimum policyCE among
checkpoints retaining native valueCE/QMAE within+0.02. Legacy core default
both-heads qualification is not the experiment gate. Both stopped on the
registered totalCEpatience, all saved evaluation checkpoints stayed valid.

| Arm | Last update | Selected | PolicyCE | ValueCE | QMAE | Whole charged wall |
| --- | --- | --- | --- | --- | --- | --- |
|control|4000|3750|2.622843894|0.664705011|0.335200605|457.804916|
|width|3000|2250|2.623800358|0.662479715|0.335335698|623.207433|

Widthgain0.008736591<0.05;
width-versus-controlgap-0.000956464<0.03.
This width/training-budget hypothesis is unsupported, not a proof that wider
networks never work. Control4000/width3000updates, total448000sampled rows;
unequal update counts follow the same registered patience rule. Both below
1800s whole-arm budget including repeated input preparation/process setup.
No isolated training-throughput ratio; existing4CPUquota/16GiB/noGPU/no paid resources.

All30native checkpoints really loaded:17control/13width, checksum/input replay,
cursor and every saved optimizer tensor verified. Sampling reconstructed from
seed17 matched TorchRNG at13common checkpoints0..3000. New stem channels changed
in actual training. AuditV1 incorrectly assumed every registered parameter gets
Adam state. The historical21parameter auxiliary material head is not used by
the policy/WDL forward, so its two tensors correctly have no optimizer state
and remain bitwise unchanged. Each nonzero-step checkpoint has46Adam states
over48registered tensors. V1source/log preserved, V2 checks those inactive
tensors plus exact optimizer state; no training or old model was modified.
Reported model parameter counts include that legacy21parameter auxiliary head.

Fixed18real histories, eager legal-masked B1/CPU1,10warmup rounds then200rounds,
alternating model order and3600timed calls/arm. Median width/control
**1.663772** (limit2.0 passed), controlmedian
0.699966ms,
widthmedian1.164584ms.
Setup/warmup0.406184s excluded;
outer measurement7.085786s includes finite
checks outside timed calls. No concurrent training, encoding/search/game latency
excluded; inference-only speed is not whole-game throughput. A latency pass
does not overturn the failed learning gate.

The24unseen36ply arena roots were frozen before transfer/update but not used
for matches after qualification failed. First preflight generator used the
wrong28plyparent; unused output/script/log preserved. Corrected book uses the
registered32plyvalue parent plus4uniformlegal suffix moves, no model/engine filter.
510tests/zero skips44.26s before production updates; real Torch and MLXCPU width
and complete guard-state integration tests passed. No Applehardware claim.

Selected controlSHAeae50a5997bf639218600a803500690efb81a21e8c0ad08137196822c3e2b6e2,
selected widthSHA41d266b48f78c4cf94f8a08b2fef822221f1d1f834dc285a1fef97bf7e178e8b.
Complete models/optimizers/input data locally retained separately from
[exact text evidence](UFUK-width-evidence-20261004.json). GitHubRelease binary
upload remains400BadContentLength; no completed remote binary-backup claim.
The prior2149file verified archive predates this width experiment; a dated
supplement is required. New Q-range normalization hypothesis has a separate
preregistration; this failed width result is preserved without a lucky extension.
