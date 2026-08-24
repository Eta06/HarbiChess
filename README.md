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
uv run harbichess-dashboard --demo
```

The network benchmark executes the actual history-aware 104-plane board input
through the MLX residual policy/WDL model. End-to-end self-play concurrency will
be calibrated separately once MCTS and the shared inference queue are present.

The standalone dashboard listens on `http://127.0.0.1:8765` by default. It
reads a low-frequency atomic telemetry snapshot and never imports or blocks the
trainer. Use `--host 0.0.0.0` to view it from another device on the local
network. Resume metadata links model, optimizer, replay cursor, RNG state,
counters, and accumulated training time so a stopped run can continue from its
latest durable checkpoint.

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
