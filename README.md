# HarbiChess

HarbiChess is a neural-guided chess engine that learns primarily through
reinforcement learning and self-play. Its first-class training target is Apple
Silicon through MLX, while chess rules, search, data formats, and training
contracts remain backend-independent.

## Architecture

```text
Chess state -> deterministic rules -> policy/WDL network -> MCTS -> move
                                      ^                 |
                                      |                 v
                                  training <- replay <- self-play
```

The rules engine decides legality and terminal results. The network learns move
priors and win/draw/loss values. Search improves those estimates, and self-play
produces the training targets.

## Development setup

HarbiChess uses `uv` and requires Python 3.12 or newer.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run harbichess-mlx-smoke --size 512 --iterations 5
uv run harbichess-network-benchmark --batches 1,8,16,32,64,128
uv run harbichess-search-benchmark --games 16,32,64,128 --simulations 32
uv run harbichess-dashboard --demo
uv run harbichess-ocak-sanity --run-id ocak-sanity-001
```

The dashboard frontend lives in `dashboard-ui` and is compiled into the Python
package's static directory. Rebuild it after frontend changes:

```bash
cd dashboard-ui
npm install
npm run build
```

The network benchmark executes the actual history-aware 104-plane board input
through the MLX residual policy/WDL model. End-to-end self-play concurrency will
be calibrated separately once MCTS and the shared inference queue are present.

The search pipeline now uses legal-masked PUCT, independent per-game random
streams, and a shared inference worker that batches parallel actors onto MLX.
Self-play records visit-policy targets and final outcomes from each position's
side-to-move perspective.

The standalone dashboard listens on `http://127.0.0.1:8765` by default. It
reads a low-frequency atomic telemetry snapshot and never imports or blocks the
trainer. Use `--host 0.0.0.0` to view it from another device on the local
network. Resume metadata links model, optimizer, replay cursor, RNG state,
counters, and accumulated training time so a stopped run can continue from its
latest durable checkpoint.

Candidate quality is tracked with color-balanced arena W/D/L results, estimated
Elo gain, and a 95% confidence interval; training loss alone never promotes a
model. See [the model quality measurement notes](docs/model-quality.md).

The OCAK learner foundation adds versioned checksummed replay, whole-game
train/validation isolation, collapse metrics, finite-gradient training, a small
pilot gate, and exact model/optimizer/sampler checkpoint resume. See
[the OCAK training guardrails](docs/ocak-training-guardrails.md).

The sanity runner streams self-play, learner, diversity, replay, and checkpoint
state to the same low-overhead dashboard snapshot. It writes immutable run
artifacts under `artifacts/runs/<run-id>` and never promotes its candidate.

Generated checkpoints, replay shards, and run artifacts are intentionally kept
outside Git history. Evaluated checkpoints will be tied to an exact source
commit and published as GitHub Release assets.

## Development phases

1. Deterministic environment and state contracts
2. MLX policy/WDL network and board encoder
3. Neural-guided MCTS with batched inference
4. Parallel self-play and replay storage
5. Training, checkpointing, and evaluation league
6. Diversity monitoring and calibrated difficulty control
