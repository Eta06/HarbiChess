# UFUK CPU inference compilation: before experiment

Question: can Inductor fuse the compact pairwise inference graph enough to reduce
actual CPU dispatch overhead while preserving logits and W/D/L semantics?
Reference mechanism: official PyTorch compile/CPU backend, conventional compiler
optimization, not an original chess algorithm or a strength gain.

Frozen model UFUK step9250 SHA
`18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae`.
Same Linux4CPU/16GiB, FP32/oneTorch thread; no GPU, paid resources or dependencies
reinstalled. C++compiler exists. Compile one process/one compiler worker, max180s
initial setup; report setup wall/RSS. Compiler caches stay in work/, no home writes.
No compiled inference is inserted into the running self-play phase.

Prepare graph only while actors run if useful; **timed eager/compiled comparisons
wait until actors/training/validation drain**. Record overlap for setup resources.
Scratch implementation first; integrate an explicit default-off mode only if the
fixed gates pass. Failure remains recorded, no silent eager fallback.

Use fullgraph Inductor/dynamic requested-action width for masked inference. Test
real opening, black-to-move, castling, en-passant and promotion inputs; selected
legal logits and WDL logits/probabilities must match eager at absolute/relative
2e-5, finite outputs, canonical actions/encoder unchanged. No bitwise promise.
A compiler exception/parity/memory/setup-budget failure rejects the experiment.

Warm each graph before measurement; include host input/output and mask gathering,
exclude board encoding as in earlier backend benchmarks. At least3paired repeats
of30calls forbatch1/8/32, oneTorch thread, order fixed byseed20261007, weights held.
Save raw timings/sources/hardware/compilation settings. Report median/p95 and
warmed runtime separately from setup cost; no inference of complete-game speed.
Qualification: batch1 median throughput improves>=15%, batch8/32 no>10%regression,
all numerical parity checks pass. If qualified, a **later registered** startup-
inclusive actor/game test is required before claiming self-play acceleration.
Do not credit the active policy learning experiment with later compile speed.
