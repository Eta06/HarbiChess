# UFUK: lossless prepared data for CPU-scale learning

Before implementation/benchmarks/new updates. This is infrastructure, not a
claim of stronger chess or successful self-learning. Previous failed model/search
experiments remain failed; the separately pinned joint experiment is unchanged.

Hypothesis: canonical bitboards/ragged legal-policy targets and memory-mapped
immutable arrays can preserve exact104history inputs/losses/sampling while
reducing repeated setup and memory enough for materially larger datasets.
One million dense FP32 inputs alone require26624000000bytes, beyond16GiB,
before policy/mask targets. Existing native panel preparation repeatedly rebuilds
histories; complete immutable80511native source stays authoritative/preserved.
Standard bit packing/mmap are not novel algorithms.

Prepared schema1:8history frames x12STM-canonical uint64piece bitboards,
8FP32metadata planes with EP encoded separately as canonical0..63/sentinel64;
EP plane is not replaced by its mean. Repetition/rule50/castling/STM exact.
Ragged legal4672action IDs and soft policy targets, STM W,D,L, source split/family/
position-key provenance. Framework-neutral NumPy decode yields exact NHWC104
FP32 tensors for Torch and MLX. Strict schema/shapes/dtypes/finite/action legality,
source-game/dataset/array hashes, immutable publish/no overwrite/no partial cache
accepted. Existing raw dataset/replay/checkpoint versions and loaders retained.
Partial historical FENs are not complete replay; do not fabricate missing clocks
or history and call it restored truth. Lichess bulk data is research only here.

Prepare80511native rows once with every source row validated and every packed
decoded input compared exactly to original BoardEncoder, including full histories.
Same source order, training-position overlap removal, whole-family split/cap
selection random.Random(seed:split); same sampled Torch indices. Native caps
40000/20000/seed24, original18f/CPU1FP32/AdamW1e-4/B64. No target changes.
Tests on EP/castling/promotion/black orientation/repetition/rule50, real Torch/
MLXCPU outputs/losses and exact next-update within-source native resume. Refuse
corrupt cache/source/schema and mixed legacy/cached run profiles. Weights-only
conversion is not resume; old runs continue under their pinned original profile.

After other training finishes, compare FOUR fresh-process preparation+100update
invocations for uncached and cached panels, alternating mode order by round,
fixed seeds24+round, same initial weights/data/caps/indices/losses. Include source
verification, process/framework/model/optimizer setup, checkpoint serialization,
native validation and whole wall/RSS. Cached total additionally charges the
ONE full80511row preparation/parity cost, even if done during earlier training.
Uncached/cached final model+optimizer+sampling states must match bitwise per
round. Do not count isolated mmap load time as total throughput.

Budget: preparation600whole-wall seconds/1CPU, eightbenchmarkinvocations
300whole-wall seconds each, no concurrent training/benchmark. Gate exact parity/
native fullresume/corruption refusals, amortized cached total<=0.50uncached total,
peak process RSS<=0.50uncached peak. Record preprocessing break-even and actual
batch-decode/update overhead. If gate fails retain evidence/default unchanged;
no lucky budget extension. If useful, the optional prepared path can support
a separately preregistered broader-data teacher/self-play experiment. No new
paid resources/storage destinations; existing4CPUquota/16GiB/32Gdisk only.
