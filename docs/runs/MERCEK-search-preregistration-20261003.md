# MERCEK: search budget diagnostic preregistration

2026-10-03. Written before any MERCEK arena. User authorized research and bounded
experiments within the existing allocation. No new paid resource or promotion.

## Question and hypothesis

Does increasing Full Gumbel search from 16 to 128 simulations, without learning
or changing the network, improve this frozen model against a weak Stockfish
control? Primary contrast: 128 minus 16, paired by opening and color.
64 is a descriptive intermediate arm. More search is not assumed to improve
an inaccurate value network. This is a small diagnostic, not a final strength gate.

## Frozen control and data

- Production source before this documentation: `972aeea94bd0cf724fee58845fe18751248eeb17`.
  Each arena records its actual starting HEAD. No search code changes in this stage.
- Initial PORT MIHVER weights: `artifacts/port-loop-20261002/checkpoints/generation-000000/model.safetensors`;
  SHA-256 `a155be94c96227815a2cac6f235c06622a5fb1df93a215b4ba391be500aaaa61`.
  1,256,355 parameters. PORT latest is not substituted after seeing results.
- Same existing eight `portable_arena.OPENINGS` families, both candidate colors;
  16 games per arm. These openings were used in PORT and are development diagnostics,
  not a newly independent final holdout. Seed `20261003`, max total ply `192`.
- Full Gumbel scales/defaults unchanged; deterministic Gumbel=0, considered actions=16.
  Only simulation count changes: run order 16, 64, 128. No seed sweep or best-run selection.
- Stockfish 19 universal binary, SHA-256
  `0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19`;
  one thread, Hash=16 MiB, 32 nodes/move, no pondering. The node limit is a weak
  control, not full-strength Stockfish and not equal compute to neural simulations.

## Compute and stop

Linux CPU FP32, PyTorch 2.14.1+cpu, four-core cgroup quota, 16 GiB memory, no CUDA;
one neural compute thread. Sequential arms, no concurrent training/testing.
Each arm: 180-second arena wall cap; external process timeout 200 seconds;
at most 600 seconds total arena process wall. Terminate on timeout, illegal
move, nonfinite output or process failure; preserve logs and any partial evidence,
report incomplete rather than retrying with changed budgets. No long training.

Exact command template (run once for each listed N; Python API used because
the existing CLI has no seed argument):

```bash
.venv/bin/python - "$N" <<'PY'
import json, sys
from pathlib import Path
from harbichess.evaluation.portable_arena import arena
n = int(sys.argv[1])
result = arena(
    Path('artifacts/port-loop-20261002/checkpoints/generation-000000/model.safetensors'),
    opponent='stockfish',
    stockfish=Path('/workspace/work/harbichess/stockfish/stockfish-linux-x86-64-universal'),
    simulations=n, nodes=32, max_plies=192, opening_pairs=8,
    wall_seconds=180, seed=20261003, threads=1,
)
out = Path(f'artifacts/mercek-search-20261003/arena-{n}.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + '\n')
PY
```

## Measurement and interpretation

Save complete moves, color/opening pairing, observed outcome vs capped games,
model/engine/source hashes, elapsed wall, CPU seconds and RSS, evaluated positions,
and the existing paired bootstrap/Hoeffding intervals. Unknown caps count 0.5
only in descriptive arena scoring; they never become terminal training labels.

Primary effect is the mean of the eight differences in opening-pair scores
(128 minus 16). Report paired bootstrap (10,000 replicates, seed 20261003) and
a conservative Hoeffding 95% interval for differences bounded in [-1,1].
Bootstrap is conditional on this fixed small suite and can degenerate. A positive
point estimate alone does not confirm strength improvement; formal support
requires the conservative primary interval above zero and no cap increase.
Report 64 independently of the primary outcome; do not use it to change the hypothesis.

Elapsed time is the measured price of increasing search, not an equal-wall-clock
strength comparison. Positions/s is not chess strength. A future adaptive-budget
controller needs an equal-time, independent-family experiment and a cheap entropy/
gap heuristic control. No Elo, calibrated probability, originality, stronger-than-
teacher or publication-quality result is established by these 48 games.

Raw artifacts remain outside Git, with hashes and compact results in the run
report. The existing PORT Release asset HTTP 400 blocker remains unresolved;
local preservation is not remote backup. Failed/inconclusive results remain recorded.

Pre-launch correction: `/usr/bin/time` is absent. The first measurement launcher
failed before starting Python/Stockfish or producing any game. Preserve that
failure; use Python `resource.getrusage` (self/children) and `time.monotonic`
instead. Seed, arms, model, stop budgets and outcome criteria stay unchanged.
