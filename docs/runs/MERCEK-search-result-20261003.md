# MERCEK frozen search budget diagnostic — 2026-10-03

## Result and decision

Completed, hypothesis **not supported**. Against Stockfish 19 limited to 32
nodes/move, 128 simulations took 9.3821× the 16-simulation arena wall time with
the same 0.125 score. No wins in any arm. Primary paired difference 128−16 is
0.000; bootstrap 95% [-0.15625, 0.1875], conservative Hoeffding 95%
[-0.960323, 0.960323]. Eight development opening pairs are underpowered;
this is not proof of equivalence or that more search never helps.

No model update, long training or champion promotion. Next work is a frozen
teacher/action-quality and value-learning diagnostic, then controlled learning/
speed experiments. Adding a large external decision model or increasing search
automatically is not supported by this result.
[Laya mechanism and research decisions](../research/MERCEK-decision-models-20261003.md).

## Preregistered controls and source

[Preregistration](MERCEK-search-preregistration-20261003.md) committed before
games; same weights, opening histories/colors, seed 20261003, total ply cap 192,
Full Gumbel=0, considered actions=16; arms ran once in order 16, 64, 128.
Existing eight PORT openings were reused and are not independent final holdouts.
Stockfish one thread/Hash=16 MiB/nodes=32, neural CPU FP32/one compute thread.
Neural simulations and Stockfish nodes are different cost units.

All three starting source HEADs:
7831489c8d6d3da362e35d31d0448bfefcf478d2. Production code unchanged from
972aeea94bd0cf724fee58845fe18751248eeb17; intervening commits are preregistration.
Source HEAD captured before play. Model generation-000000 SHA-256:
a155be94c96227815a2cac6f235c06622a5fb1df93a215b4ba391be500aaaa61.
Stockfish binary SHA-256:
0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19.
Hashes independently checked in every arena report. No new data split/label
schema or MLX conversion. Old checkpoint/replay/source results preserved.

## Measurements

W/D/L is candidate wins/draws/losses. All 48 games ended in observed terminal
outcomes; no max-ply caps and no relabeled unknown outcomes.

| Simulations | W/D/L | Score | Pair bootstrap 95% | Arena wall s | Process wall s | Neural evaluations |
|---:|---:|---:|---|---:|---:|---:|
| 16 | 0/4/12 | 0.12500 | [0.03125, 0.25000] | 11.4522 | 12.8493 | 5,796 |
| 64 | 0/5/11 | 0.15625 | [0.03125, 0.28125] | 55.4385 | 56.7099 | 28,399 |
| 128 | 0/4/12 | 0.12500 | [0.03125, 0.25000] | 107.4457 | 108.6939 | 51,548 |

Opening-pair scores:

- 16: [0.50, 0, 0, 0.25, 0, 0, 0, 0.25].
- 64: [0.25, 0, 0, 0.25, 0, 0.25, 0.50, 0].
- 128: [0.25, 0.25, 0.50, 0, 0, 0, 0, 0].
- Primary pair differences: [-0.25, 0.25, 0.50, -0.25, 0, 0, 0, -0.25].

Bootstrap: 10,000 resamples of eight paired differences, seed 20261003.
Hoeffding radius sqrt(2·log(40)/8) for bounded [-1,1] differences.
The 64 arm's small positive point difference is descriptive, not a selected
winner or evidence of improvement. No Elo inference.

Total process wall 178.2530 s; summed arena wall 174.3364 s; self+children
CPU 176.3508 s; 85,743 neural position evaluations. Python resource.getrusage
recorded user/system CPU and per-process high-water RSS, not aggregate peak
allocation memory. Largest self RSS 264,612 KiB (~258.4 MiB); children high
water 324,300 KiB. No training/testing ran concurrently with arenas; only light
source reads/downloads. Run order was frozen but not randomized; residual host
noise and temperature/order effects were not measured. These are measured costs
of fixed simulation budgets, not equal-wall strength controls.

Allocation rechecked: cgroup cpu.max 400000/100000 (4 CPU), memory.max 16 GiB;
5 visible CPUs, PyTorch 2.14.1+cpu, CUDA unavailable. No new paid resources.
Currency tariff unavailable; no invented currency cost.

## Commands, execution errors and verification

Exact arena API invocation is in preregistration; executed through preserved
run-search-arm.py with a 200-second subprocess timeout and unchanged 180-second
arena cap. All return codes 0, each below the caps, total below 600 seconds.
No arena retry, alternate seed, weight substitution or budget change.

Pre-launch GNU time wrapper was absent: launcher failed before any arena
process/game. Recorded before play and replaced with stdlib resource timing.
The existing arena CLI has no seed argument, so seed was passed explicitly to
the existing Python API before play. Independent game verifier initially used
system Python without chess; corrected to the existing .venv without replaying
the experiment. These setup errors are preserved, not counted as chess failures.

All 48 saved complete move histories replayed through python-chess: every move
legal, exact opening prefix and ply count, terminal outcome and candidate-color
score agree. Seed/model/engine budget/hash fields checked. Existing regression:

    .venv/bin/pytest -q tests/test_torch_search.py tests/test_full_gumbel_search.py tests/test_full_gumbel_targets.py tests/test_rules.py

**25 passed in 1.44 s, zero skipped/failed**. This stage changes documentation
only; previous PORT full 461-pass result is historical, not a new full-suite run.
Apple Metal and CUDA devices remain untested; no new parity claim.

## Artifacts and reproducibility limits

Local raw records: artifacts/mercek-search-20261003, including full games,
summaries, resource use, exit logs, validation receipt, script and manifest.
Source snapshots: /workspace/work/harbichess/research-20261003, URL/hash metadata
in the [Git source inventory](../research/MERCEK-sources-20261003.json).
The existing PORT Release upload-path HTTP 400 issue remains unresolved.
MERCEK's small text-only evaluation records are also preserved in the
[Git evidence bundle](MERCEK-search-evidence-20261003.json): 14 UTF-8 members
with original bytes/checksums, including all 48 games. This is not model or
training-replay binary publication. Restore to a new empty directory using the
bundle instructions; every embedded member hash was verified locally.
Publication and remote readback are reported in the closing commit inventory.
Full downloaded literature snapshots remain local; URLs/hash receipts are in Git.
Local files were not removed. PORT model/replay Release backup remains incomplete;
this small textual evidence fallback does not close that separate gap.
Archival deviation from preregistration: after completing the unchanged experiment,
the 83,581-byte textual bundle was selected for Git to preserve the full small
arena evidence despite the Release blocker. Experiment arms, seeds, thresholds
and analyses were not altered. Large model/replay artifacts remain outside Git.

| Raw record | SHA-256 |
|---|---|
| arena-16.json | dcd6363d40d206a253dc51d5e5795b595d9f0e36d37531abd0b09fe91085b14d |
| arena-64.json | a8d5b4a1e14ac48a01f0f5fa870f9c1e0e7d748edbeff1f673e4fac448151d8d |
| arena-128.json | 1cc7cb68ee631716809eca663d0b2299e8cbab623ab438f02aef9e7e54231a7c |
| summary.json | a5da50e3c0496fa05791add9880966206f35c45d437842ae7a9e626d05b7835d |

The inspected Laya model was not run or trained. Its developer latency/
classification/calibration metrics are not local chess measurements.
Novelty and stronger-student paths remain hypotheses documented separately.
