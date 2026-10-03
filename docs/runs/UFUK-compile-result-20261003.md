# UFUK CPU compilation: measured result

The preregistered warmed inference gate **passed**. This is not a complete-game
or self-play speed result, nor a chess strength gain. Eager remains the default;
actors, training and MLX are unchanged. Explicit `TorchPolicyValueBackend(network,
compiled=True)` enables CPU pairwise inference only. Dense models and unsupported
devices are refused. There is no silent fallback on compilation failure.

## Failed attempt preserved

[V1](UFUK-compile-preregistration-20261003.md), source
`ba30232`: fullgraph compilation of the public masked method failed on its
data-dependent action-bounds guard. Setup wall2.4298s/CPU1.9786s/RSS345828KiB;
zero parity cases completed. `artifacts/ufuk-compile-20261003/setup.json` and
`failure.txt` remain intact. Bounds checking was not removed or bypassed.

[V2](UFUK-compile-preregistration-v2-20261003.md), source
`43376c35d74a2340773f83a623135e4e0480dc9e`: compile only private `_features`,
retain shape/range checks on the host, including their cost in the timed backend
boundary. Frozen UFUK step9250 SHA
`18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae`;
PyTorch2.14.1+cpu/Inductor/fullgraph/dynamic/FP32/one inference and compiler thread.
Existing4CPU/16GiB allocation, no GPU or new paid resources.

## Numerical and speed evidence

V2 setup35.942599s, parent CPU9.780154s, peak parent RSS462420KiB. Compiler child
CPU is not included in that parent counter. All18 FEN×batch cases passed at2e-5
tolerance: initial, both castling colours, both promotion colours, en passant;
B1/8/32. Maximum logged policy error1.32e-6, WDL error4.77e-7.
Setup overlapped the late self-play generation; it is not a speed measurement.

The isolated warmed benchmark ran after actors/validation drained. Three repeat
blocks×30 calls/arm/batch, randomized arm order seed20261007;540 raw timings.
Host input/output conversion and host guards included; board encoding, search,
compiler setup excluded. Counts are evaluated positions, not played moves.

| Batch | Eager positions/s | Compiled positions/s | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 1085.1571 | 1467.7822 | 1.352599 |
| 8 | 2277.3368 | 2565.3425 | 1.126466 |
| 32 | 2667.7398 | 2924.4178 | 1.096216 |

B1 median batch latency0.891894→0.659310ms, p95 1.110368→0.773161ms.
Registered B1≥15% and B8/B32 regression≤10% gates passed. These local medians
have no multi-machine uncertainty estimate. Cold startup35.94s must be charged
before any complete-game speed claim. Small batch performance does not prove
four-process self-play gains; that comparison has not been run.

The benchmark separately checked logits and soft probabilities at2e-5 and refused
three invalid action requests. The integrated adapter is commit`65e7627`, its
real Inductor mixed-board/individual parity and malformed-action test`c1fb105`:
`tests/test_torch_compiled.py`,1 passed15.74s, no skips. That integration uses a
tiny real model; performance above uses the fixed trained72,497-parameter model.
Apple Metal/CUDA compilation is not claimed or tested here.

## Reproduction and limitations

Exact setup/benchmark scripts and results are stored with SHA-256 in
[the text evidence](UFUK-compile-evidence-20261003.json). Scripts use
`TORCHINDUCTOR_CACHE_DIR=/workspace/work/harbichess/inductor-cache` and
`TORCHINDUCTOR_COMPILE_THREADS=1`; setup/benchmark command receipts are included.
The compiler cache is disposable, not a model checkpoint. Model bytes remain
locally preserved. GitHub Release transport still returns400; no completed remote
binary backup is claimed. Text evidence in Git is not a substitute for the weights.

During the separate original strength-arena sequence, this optional adapter was
edited in the shared source tree. Default math stayed eager, but declared commit
alone did not identify the dirty module. The flaw is disclosed in
[the source-integrity verification protocol](UFUK-arena-provenance-verification-20261003.md).
Neither speed nor arena results should conceal that provenance problem.
