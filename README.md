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
```

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

