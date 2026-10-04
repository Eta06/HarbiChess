# UFUK: lossless prepared native data result

Infrastructure gates **PASSED**. Main strong/fast/reliable self-learning and
Stockfish/AlphaZero-level goal remains unmet. This experiment changes data
storage/loading, not chess knowledge or target strength. Previous joint-context,
sparse-value/search/width and self-learning failures remain failed.
[Prospective protocol](UFUK-packed-data-preregistration-20261004.md).

Clean pinned source `f1dc217ab0f8c22b02acc54f2e706ef400145498`. Optional packed schema1/prepared
schema1;8history frames x12STM-canonical uint64piece bitboards,8FP32metadata,
separate full-square EP including ghost EP, ragged legal/action targets and
STM W,D,L. Decode exact NHWC104FP32. Framework-neutral NumPy reader; explicit
Torch batch adapter; actual MLXCPU prepared loss/gradient/update tested. Metadata
is not a FEN-history guess. Existing raw80511native source/old replay and checkpoints
unchanged. Standard bit packing/mmap are not claimed as novel algorithms.

Original80511native teacher rows/1024games/48train16validation families:
datasetSHA b34a3870b494b4cc0fe8a1df87960fe5e2406a5c9a2ed341d1b93fed0e0e094f.
Source file/dataset/metadata hashes, row legal/policy/WDL/provenance and EVERY full
packed input compared bitwise to authoritative BoardEncoder at preparation.
All80511rows passed. Packed12arrays total81145542bytes,
immutable SHA manifests/fsync/no overwrite/no incomplete publication accepted.
Whole fresh preparation/import process128.375422s;
internal127.083460s starts after module import, NOT used as whole cost.

Original18f model/CPU1FP32/Torch2.14.1+cpu/deterministic/AdamW1e-4/wd1e-4/
clip5/B64/caps40000/20000/max100/eval100/patience1000, four seeds20261024..27.
Same source order, training-position overlap removal, whole-family caps and
indices. Eight fresh invocations alternate uncached/cached order; each300whole
wall seconds including import/framework/model/optimizer/setup/source/cache hashes,
two20000row validation passes and native checkpoints at0/100. No concurrent
training/benchmark. Preparation600whole-wall budget; no timeout extension.
Total800actual updates/51200sampled rows. No strength arena/promotion.

| Seed | Uncached whole seconds | Cached whole seconds | Uncached peak KiB | Cached peak KiB |
| --- | --- | --- | --- | --- |
|20261024|132.206835|15.966451|4336312|430056|
|20261025|131.363238|15.660765|4336556|430880|
|20261026|132.089976|15.767464|4336340|429464|
|20261027|131.852374|15.963350|4336280|430352|

Uncached total **527.512423s**; cached includes ONE
full preparation plus four invocations **191.733452s**,
ratio **0.363467178<=0.50**. PeakRSS ratio
**0.099359953<=0.50**. Both preregistered cost gates pass;
measured preprocessing break-even2invocations.
These are short100update setup-inclusive invocations, not an8x long-training
optimizer throughput claim. Entire driver719.251544s.
Linux mmap/RSS depends on filesystem/runtime/data scale; one host/four seeds
are not hardware-general statistics or proof of million-row throughput.

All16actual native checkpoints loaded, verifying complete model+Adam states,
Torch RNG/cursor/run profiles/input hashes. Paired initial/final model and every
optimizer/RNG/validation/run-state value BITWISE equal. Reconstructed100x64
sample indices each seed match RNG exactly. Raw1026verified input paths retained;
cached1039adds12arrays+manifest. Profile differs only by versioned prepared data
manifest/source code hashes. Mixed oldraw/newcached resume refuses; old runs must
resume with original pinned source/profile/runtime/input paths. Optional data
conversion is not optimizer migration, weights-only transfer or full resume.
Step0noAdam states; step100unused historical material auxiliary legitimately has
no Adam state. Audit2.440779s, no extra gradient updates.

Supplementary exact selection-only probe1000alternatingB64batches: median
packed0.854638ms vs dense
0.289765ms. Dense comparator is512resident
rows decoded from same source, not full40000row locality/RSS. Every target/input
batch exactly equal, no optimizer updates; isolated select probe is not main
cost gate. Decoding has CPU overhead; memory/setup savings dominate this measured
short-run task. Update-only time was not separately instrumented.

Before production **553passed/zero skips45.45s**, targeted6passed2.42s, Ruffpass.
Full histories/Black/castle/promotion/EP/ghostEP/repetition/rule50/big-endian uint64,
real Torch/MLXCPU inference/loss/gradient/update, cached whole vs split native
optimizer/RNG/next-update resume, raw/cached exact updates and corruption refusal.
Intermediate E501format fixes and initial test failure retained: native loader
requires expected_run_config; TEST omitted it and was corrected, checkpoint guard
not weakened. Previous552passed45.23s occurred BEFORE adding actualMLX update test;
553suite rerun was justified. AppleMetal/CUDA unavailable, untested; no device
test skips. Backend AdamW defaults differ, no cross-framework optimizer equivalence.

Author+committer Emir Tunahan Alim <emrtnhalim@gmail.com>, pushEta06/admin, no
coauthor/onefilepercommit. Seven source/protocol files actually downloaded from
GitHub immutable f1dc217andSHAchecked. Joint result85UTF8 remote restored/verified.
Space20important completion/start readback exact;84prior blocks unchanged, old
heading/history preserved. New local cache/full native state binary artifacts
await verified fifth archive supplement. Release upload400BadContentLength remains
blocked; Git text evidence does not complete remote binary model/data backup.

Optional cached input can now support a SEPARATELY preregistered broader-history
teacher/self-play study on existing4CPU/16GiB/noGPU. No larger collection/training
or stronger model was performed in this experiment. No new paid resources.
