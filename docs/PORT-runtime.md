# Linux execution and checkpoint operations

The existing environment needs only missing extras; do not rebuild it unnecessarily.
For a fresh checkout, `uv sync --extra dev --extra torch --extra parity` installs the locked
Linux PyTorch CPU and MLX CPU parity packages. Apple Silicon keeps its normal MLX dependency;
install the torch extra too when running the complete test suite. `parity` is Linux-only.
No NVIDIA device is assumed. CUDA requires an explicitly appropriate PyTorch installation
instead of the Linux CPU index; check `torch.cuda.is_available()` before `--device cuda`.
The CUDA path has not been device-tested in this allocation.

Full verification (no new skips):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q
.venv/bin/ruff check .
```

Linux MLX CPU executes legacy tests; it does not verify Apple Metal/BF16 performance.
Loss/gradient parity is tested. PyTorch AdamW is bias-corrected; historical MLX AdamW is not.
Do not compare equal-step training as equivalent algorithms or reuse optimizer states across them.

## Restore and warm start

Follow `docs/research/README.md` to verify the old Release SHA256SUMS and member manifest.
Extract to a separate directory. Keep originals; do not overwrite archived provenance.

```bash
harbichess-convert-mlx /path/to/old/model.safetensors artifacts/converted/model.safetensors --legacy-mlx
harbichess-torch-loop artifacts/new-run --weights artifacts/converted/model.safetensors \
  --generations 3 --games 8 --simulations 16 --steps 32 --threads 1 --workers 4 --wall-seconds 1800
```

The converter supports base, invariant, decoupled MIHVER and plastic DENGE networks.
Unknown heads (including spatial-policy/action-value experiments), keys or shapes fail strictly.
`--export-mlx` emits MLX OHWI weights with the same production head structure; instantiate the
matching MLX class/config before loading. Safetensors metadata carries the specification and
source SHA. This is version 1 model transfer and a fresh optimizer, **not full training resume**.
Existing MLX checkpoint/replay files remain readable by their original code. The replay loader
accepts its existing supported target versions; encoder/action/STM semantics do not change.

## Full PyTorch training resume

```bash
harbichess-torch-loop artifacts/new-run \
  --resume artifacts/new-run/checkpoints/generation-000001 --generations 3 \
  --games 8 --simulations 16 --steps 32 --threads 1 --workers 4 --wall-seconds 1800
```

All loop options except the final generation target and session wall budget must match the saved
config. Runtime version, device, threads and deterministic setting must match for exact resume.
Checkpoints include model, AdamW state, torch/CUDA RNG, sampler RNG, immutable replay checksums,
learner step, game cursor, run identity and complete-generation history. The replay files and
checkpoint directory form a unit: relocate the complete run directory. Never copy only weights
and call it resume. Early version 1 cursors missing run_id are migrated from replay provenance.

Actors drain before learning. CPU threads share a frozen inference copy; no process fork or
concurrent learner mutation. One-worker deterministic CPU CLI resume is tested bitwise. Multiple
actor batching can change floating-point aggregation/timing, so it has seed reproducibility but
no blanket bitwise claim. A partial generation restarts from the last complete boundary; if its
regenerated replay differs, the process rejects it and keeps the earlier diagnostic bytes.

`result.json` reports the most recent session; checkpoint history stores committed generations.
Preserve each session result before a later resume when accounting total compute (the PORT
experiment includes `session-generation-1.json`). Budget exhaustion reports `budget-stopped`;
other exceptions report `failed` and raise. Use last_complete_checkpoint, not partial in-memory
learner progress. Models are latest research learners; promotion_ready stays false.

Metrics distinguish generated unique rows, sampled update rows, known outcome rows, inference
requests and self-play wall throughput. Prepared tensors are bounded by the replay window;
batch construction/validation overhead is included in session wall time, outside optimizer-step
throughput. Model strength requires paired games, capped-game counts and uncertainty, not CE alone.

## Measure inference and game strength

```bash
.venv/bin/python -m harbichess.benchmarks.torch artifacts/converted/model.safetensors \
  --output artifacts/measurements/inference.json
.venv/bin/python -m harbichess.evaluation.portable_arena artifacts/converted/model.safetensors \
  stockfish --stockfish /path/to/stockfish --nodes 32 --simulations 16 \
  --output artifacts/measurements/stockfish.json
```

Random, frozen-network and no-op controls use the same eight opening families and both colors.
The default 16-game arena is diagnostic. Caps are scored 0.5 only for that diagnostic, reported
separately, and never converted to known training draws. Pair bootstrap can degenerate on tiny
samples; a wide Hoeffding interval is also shown, conditional on independent opening families.
No general Elo or promotion claim follows from this fixed suite.
